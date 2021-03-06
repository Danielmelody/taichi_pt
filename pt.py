import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

inf = 1e10
eps = 1e-4

max_luminance = 100

class Image:
    def __init__(self, path):
        self.img = ti.tools.imread(path)
        tex_w, tex_h = self.img.shape[0:2]
        self.field = ti.Vector.field(3, dtype=ti.u8, shape=(tex_w, tex_h))

    def load(self):
        self.field.from_numpy(self.img)


@ti.func
def tex2d(tex_field, uv):
    tex_w, tex_h = tex_field.shape[0:2]
    p = uv * ti.Vector([tex_w, tex_h])
    l = int(ti.floor(p))
    t = p - l

    # bilinear interp
    return (
        ((tex_field[l] * (1 - t[0])) + tex_field[l + ti.Vector([1, 0])] * t[0]) * (1 - t[1])
        + (tex_field[l + ti.Vector([0, 1])] * (1 - t[0]) + tex_field[l + ti.Vector([1, 1])] * t[0]) * t[1]) / 255.0
    
def gaussian_kernel(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l, dtype=np.single)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig), dtype=np.single)
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


@ti.data_oriented
class SphereSOA:
    def __init__(self, array_length):
        self.center_radius = ti.Vector.field(4, dtype=ti.f32, shape=(array_length))
        self.albedos = ti.Vector.field(3, dtype=ti.f32, shape=(array_length))
        self.emissions = ti.Vector.field(3, dtype=ti.f32, shape=(array_length))
        self.roughness = ti.field(dtype=ti.f32, shape=(array_length))
        self.metallics = ti.field(dtype=ti.f32, shape=(array_length))
        self.iors = ti.field(dtype=ti.f32, shape=(array_length))
        self.array_length = array_length

    @ ti.func
    def intersect(self, o, d):
        min_t = inf
        min_index = -1
        sp_index = 0
        for i in ti.ndrange(self.array_length):
            c_r = self.center_radius[i]
            r = c_r[3]
            p = ti.Vector([c_r[0], c_r[1], c_r[2]])
            op = p - o
            b = op.dot(d)
            det = b * b - op.dot(op) + r * r
            t = inf
            if det > 0:
                det = ti.sqrt(det)
                t = (b - det) if ((b - det) > eps) else ((b + det)
                                                         if (b + det > eps) else inf)

            if t < min_t:
                min_index = sp_index
                min_t = t

            sp_index += 1
        return ti.Vector([min_t, min_index])


width, height = 1080, 720


linear_pixel = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
errors = ti.field(dtype=ti.f32, shape=(width, height))
ray_count = ti.field(dtype=ti.i32, shape=(width, height))
bloom_pixel = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
pixel = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

gaussian_kernel_size = 9
bloom_strength = 0.02
ambient_weight = 0.2
start_cursor = [0.0, 0.0]



blur_kernel = ti.field(ti.f32, shape=(gaussian_kernel_size, gaussian_kernel_size))
blur_kernel.from_numpy(gaussian_kernel(gaussian_kernel_size, 1.))

skybox = Image('skybox.jpg')  # z > 0


last_camera_pos = ti.field(ti.f32, 3)
camera_pos = ti.field(ti.f32, 3)
focal_length = ti.field(ti.f32, shape=())

class SphereAOS:
    def __init__(self, center_radius, albedo, emission, roughness, metallic, ior):
        self.center_radius = center_radius
        self.albedo = albedo
        self.emission = emission
        self.roughness = roughness
        self.metallic = metallic
        self.ior = ior


spheres_aos = [
    SphereAOS(
        center_radius=ti.Vector([-2.,0.0, -3.0, 2.5]), albedo=ti.Vector([1.0, 1.0, 0.0]), emission=ti.Vector([0, 0, 0]), roughness=0.2, metallic=1.0, ior=2.495),
    SphereAOS(
        center_radius=ti.Vector([1.0, -1.5, 1.0, 1.0]), albedo=ti.Vector([1.0, 1.0, 1.0]), emission=ti.Vector([0, 0, 0]), roughness=0.0, metallic=0.0, ior=1.4),
    SphereAOS(
        center_radius=ti.Vector([0.0, -500., 0., 497.5]), albedo=ti.Vector([1.0, 0.8, 0.6]), emission=ti.Vector([0, 0, 0]), roughness=0.5, metallic=0.95, ior=2.90),
    SphereAOS(
        center_radius=ti.Vector([-1, -2.0, 0.5, 0.5]), albedo=ti.Vector([1.0, 1.0, 1.0]), emission=ti.Vector([0, 0, 0]), roughness=0.0, metallic=1.0, ior=2.5),
    SphereAOS(
        center_radius=ti.Vector([2.0, 0, -2.0, 0.7]), albedo=ti.Vector([0, 0, 0]), emission=ti.Vector([15, 1, 1]), roughness=0.0, metallic=0.8, ior=2.0),
    SphereAOS(
        center_radius=ti.Vector([-2.0, 1.0, 1.0, 0.7]), albedo=ti.Vector([0, 0, 0]), emission=ti.Vector([0.2, 15, 15]), roughness=0.0, metallic=0.8, ior=2.0),
    SphereAOS(
        center_radius=ti.Vector([-8.0, -1.5, -5, 1.01]), albedo=ti.Vector([1.0, 1.0, 1.0]), emission=ti.Vector([4, 4, 0.5]), roughness=0.5, metallic=0.2, ior=2.0),
    
]

spheres_soa = SphereSOA(len(spheres_aos))

for i, sphere in enumerate(spheres_aos):
    spheres_soa.center_radius[i] = sphere.center_radius
    spheres_soa.albedos[i] = sphere.albedo
    spheres_soa.emissions[i] = sphere.emission
    spheres_soa.roughness[i] = sphere.roughness
    spheres_soa.metallics[i] = sphere.metallic
    spheres_soa.iors[i] = sphere.ior

skybox.load()


@ ti.func
def reflect(I, N):
    return I - 2 * N.dot(I) * N


@ ti.func
def lerp(vl, vr, frac):
    return vl + frac * (vr - vl)


@ ti.func
def cubemap_coord(dir):
    eps = 1e-7
    coor = ti.Vector([0., 0.])
    if dir.z >= 0 and dir.z >= abs(dir.y) - eps and dir.z >= abs(dir.x) - eps:
        coor = ti.Vector([3 / 8, 1 / 2]) + \
            ti.Vector([dir.x / 8, dir.y / 6]) / dir.z
    if dir.z <= 0 and -dir.z >= abs(dir.y) - eps and -dir.z >= abs(dir.x) - eps:
        coor = ti.Vector([7 / 8, 1 / 2]) + \
            ti.Vector([-dir.x / 8, dir.y / 6]) / -dir.z
    if dir.x <= 0 and -dir.x >= abs(dir.y) - eps and -dir.x >= abs(dir.z) - eps:
        coor = ti.Vector([1 / 8, 1 / 2]) + \
            ti.Vector([dir.z / 8, dir.y / 6]) / -dir.x
    if dir.x >= 0 and dir.x >= abs(dir.y) - eps and dir.x >= abs(dir.z) - eps:
        coor = ti.Vector([5 / 8, 1 / 2]) + \
            ti.Vector([-dir.z / 8, dir.y / 6]) / dir.x
    if dir.y >= 0 and dir.y >= abs(dir.x) - eps and dir.y >= abs(dir.z) - eps:
        coor = ti.Vector([3 / 8, 5 / 6]) + \
            ti.Vector([dir.x / 8, -dir.z / 6]) / dir.y
    if dir.y <= 0 and -dir.y >= abs(dir.x) - eps and -dir.y >= abs(dir.z) - eps:
        coor = ti.Vector([3 / 8, 1 / 6]) + \
            ti.Vector([dir.x / 8, dir.z / 6]) / -dir.y
    return coor


@ ti.func
def cosine_sample(n):
    u = ti.Vector([1.0, 0.0, 0.0])
    if abs(n[1]) < 1 - eps:
        u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    v = n.cross(u)
    phi = 2.0 * math.pi * ti.random()
    ay = ti.sqrt(ti.random())
    ax = ti.sqrt(1.0 - ay**2.0)
    return ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n


@ ti.func
def ggx_sample(n, wo, roughness):
    u = ti.Vector([1.0, 0.0, 0.0])
    if abs(n[1]) < 1.0 - eps:
        u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
    v = n.cross(u)
    r0 = ti.random()
    r1 = ti.random()
    a = roughness ** 2.0
    a2 = a * a
    theta = ti.acos(ti.sqrt((1.0 - r0) / ((a2-1.0)*r0 + 1.0)))
    phi = 2.0 * math.pi * r1
    wm = ti.cos(theta) * n + ti.sin(theta) * ti.cos(phi) * \
        u + ti.sin(theta) * ti.sin(phi) * v
    wi = reflect(-wo, wm)
    return wi


@ ti.func
def ggx_ndf(wi, wo, n, roughness):
    m = (wi + wo).normalized()
    a = roughness * roughness
    nm2 = n.dot(m) ** 2.0
    return (a * a) / (math.pi * nm2*(a*a-1.0)+1.0**2.0)


@ ti.func
def ggx_ndf_wi(wi, wo, n, roughness):
    m = (wi + wo).normalized()
    wm_pdf = ggx_ndf(wi, wo, n, roughness)
    wi_pf = n.dot(m) / (4 * m.dot(wi))
    return wi_pf


@ ti.func
def ggx_pdf(roughness, hdotn, vdoth):
    t = hdotn*hdotn*roughness*roughness - (hdotn*hdotn - 1.0)
    D = (roughness*roughness) * math.pi / (t*t)
    return D*hdotn / (4.0 * abs(vdoth))


@ ti.func
def schlick2(wi, n, f0):
    return f0 + (1.0-f0) * (1-n.dot(wi))**5.0


@ ti.func
def smith_g(NoV, NoL, roughness):
    a2 = roughness * roughness
    ggx_v = NoL * ti.sqrt(NoV * NoV * (1.0 - a2) + a2)
    ggx_l = NoV * ti.sqrt(NoL * NoL * (1.0 - a2) + a2)
    return 0.5 / (ggx_v + ggx_l)


@ti.func
def ggx_smith_uncorrelated(roughness, hdotn, vdotn, ldotn, fresnel):
    t = hdotn*hdotn*roughness*roughness - (hdotn*hdotn - 1.0)
    D = (roughness*roughness) * math.pi / (t*t)
    F = fresnel
    Gv = vdotn * ti.sqrt(roughness*roughness +
                         (1.0 - roughness * roughness)*ldotn*ldotn)
    Gl = ldotn * ti.sqrt(roughness*roughness +
                         (1.0 - roughness * roughness)*vdotn*vdotn)
    G = 1.0 / (Gv + Gl)
    return F*G*D / 2.0


@ ti.func
def luma(albedo):
    return albedo.dot(ti.Vector([0.2126, 0.7152, 0.0722]))

@ti.func
def tone(x):
	A = 2.51
	B = 0.03
	C = 2.43
	D = 0.59
	E = 0.14
	return (x * (A * x + B)) / (x * (C * x + D) + E)



gui = ti.GUI("Path Tracer", res=(width, height))


@ ti.kernel
def trace():
    for i, j in linear_pixel:
        adaptive_sampler_count = int(16 * tone(errors[i, j] / ray_count[i, j]))
        # pixel[i, j].fill(adaptive_sampler_count / 16)
        for addition_sampler in ti.ndrange(16):
            if addition_sampler > adaptive_sampler_count:
                continue
            ray_count[i, j] += 1
            sample = ray_count[i, j]
            o = ti.Vector([camera_pos[0], camera_pos[1], camera_pos[2]])
            aperture_size = 0.1
            forward = -o.normalized()
    
            u = ti.Vector([0.0, 1.0, 0.0]).cross(forward).normalized()
            v = forward.cross(u).normalized()
            u = -u
    
            d = ((i + ti.random() - width / 2.0) / width * u
                 + (j +  ti.random() - height / 2.0) / width * v
                 + width / width * forward).normalized()
    
            focal_point = d * focal_length[None] / d.dot(forward) + o
    
            # assuming a circle-like aperture
            phi = 2.0 * math.pi * ti.random()
            aperture_radius = ti.random() * aperture_size
            o += u * aperture_radius * ti.cos(phi) + \
                v * aperture_radius * ti.sin(phi)
    
            d = (focal_point - o).normalized()
            uv = cubemap_coord(d)
            albedo_factor = ti.Vector([1.0, 1.0, 1.0])
            radiance = ti.Vector([0.0, 0.0, 0.0])
            for step in ti.ndrange(32):
                sp = spheres_soa.intersect(o, d)
                uv = cubemap_coord(d)
                if sp[1] > -1:
                    sp_index = int(sp[1])
                    p = sp[0] * d + o
                    c_ = spheres_soa.center_radius[sp_index]
                    c = ti.Vector([c_[0], c_[1], c_[2]])
                    radius = c_[3]
                    n = (p - c).normalized()
                    wo = -d
                    albedo = spheres_soa.albedos[sp_index]
                    metallic = spheres_soa.metallics[sp_index]
                    ior = spheres_soa.iors[sp_index]
                    roughness = ti.max(0.04, spheres_soa.roughness[sp_index])
                    f0 = (1.0 - ior) / (1.0 + ior)
                    f0 = f0 * f0
                    f0 = lerp(f0, luma(albedo), metallic)
                    wi = reflect(-wo, n)
                    radiance += spheres_soa.emissions[sp_index] / (radius * radius) * albedo_factor
    
                    view_fresnel = schlick2(wo, n, f0)
                    sample_weights = ti.Vector([1.0 - view_fresnel, view_fresnel])
    
                    weight = 0.0
    
                    h = (wi + wo).normalized()
    
                    shaded = ti.Vector([0.0, 0.0, 0.0])
    
                    if ti.random() < sample_weights[0]:
                        wi = cosine_sample(n).normalized()
                        h = (wi + wo).normalized()
                        shaded = ti.max(0.00, wi.dot(n) * albedo / math.pi)
    
                    else:
                        wi = ggx_sample(n, wo, roughness).normalized()
                        h = (wi + wo).normalized()
                        F = schlick2(wi, n, f0)
                        shaded = ti.max(0.0, wi.dot(n) * albedo * ggx_smith_uncorrelated(
                            roughness, h.dot(n), wo.dot(n), wi.dot(n), F))
    
                    pdf_lambert = wi.dot(n) / math.pi
                    pdf_ggx = ggx_pdf(roughness, h.dot(n), wo.dot(h))
    
                    weight = ti.max(0.0, 1.0 / (sample_weights[0] *
                                                pdf_lambert + sample_weights[1] * pdf_ggx))
    
                    # russian roule
                    albedo_factor *= shaded * weight
                    if step > 5:
                        if luma(albedo) < ti.random():
                            break
                        else:
                            albedo_factor /= luma(albedo)
                    d = wi
                    o = p + eps * d
    
                else:
                    radiance += albedo_factor * (tex2d(skybox.field, uv)) * ambient_weight
                    break
            linear_color = radiance

            linear_pixel[i, j] = (linear_pixel[i, j] * (sample - 1) + linear_color) / sample
            
            errorVec = linear_color - linear_pixel[i, j]
            error = errorVec.dot(errorVec)
            errors[i, j] += error
            
            luminance = radiance.dot(ti.Vector([0.2126, 0.7152, 0.0722]))
            
            if(luminance < max_luminance):
                if (luminance > 1.0):
                    bloom_pixel[i, j] = (bloom_pixel[i, j] * (sample - 1) + linear_color) / sample
                else:
                    bloom_pixel[i, j] = (bloom_pixel[i, j] * (sample - 1)) / sample
            
    for i, j in bloom_pixel:
        hdr_blur = ti.Vector([0.0, 0.0, 0.0])
        for kx in ti.ndrange(gaussian_kernel_size):
            for ky in ti.ndrange(gaussian_kernel_size):
                hdr_blur += bloom_pixel[i + kx - gaussian_kernel_size // 2, j + ky - gaussian_kernel_size // 2] * blur_kernel[kx, ky]
        
        bloom_pixel[i, j] = hdr_blur
        
    for i, j in pixel:
        pixel[i, j] = pow(tone(linear_pixel[i, j] + bloom_pixel[i, j] * bloom_strength), 1.0 / 2.2)


def try_reset(t):
    for e in gui.get_events(gui.PRESS, gui.MOTION):
        if e.key == gui.LMB:
            last_camera_pos[0] = camera_pos[0]
            last_camera_pos[1] = camera_pos[1]
            last_camera_pos[2] = camera_pos[2]
            start_cursor_x, start_cursor_y = gui.get_cursor_pos()
            start_cursor[0] = start_cursor_x
            start_cursor[1] = start_cursor_y

        if e.key == gui.MOVE and gui.is_pressed(gui.LMB, gui.RMB):
            (current_cursor_x, current_cursor_y) = gui.get_cursor_pos()

            rotateX = (current_cursor_x - start_cursor[0])
            rotateY = (current_cursor_y - start_cursor[1])

            x = math.cos(
                rotateX) * last_camera_pos[0] - math.sin(rotateX) * last_camera_pos[2]
            z = math.sin(
                rotateX) * last_camera_pos[0] + math.cos(rotateX) * last_camera_pos[2]
            y = last_camera_pos[1]
            camera_pos[1] = math.cos(
                rotateY) * y - math.sin(rotateY) * z
            camera_pos[2] = math.sin(
                rotateY) * y + math.cos(rotateY) * z
            camera_pos[0] = x
            return True

        if e.key == gui.WHEEL:
            dt = e.delta[1] / 1000.0
            if gui.is_pressed(gui.SHIFT):
                focal_length[None] *= (1.0 + dt)
            else:
                for i in range(3):
                    camera_pos[i] *= (1.0 + dt)
            return True

    return False

# Initialize the camera
sample = 0
t = 1
focal_length[None] = 15.0
last_camera_pos[2] = focal_length[None]
camera_pos[2] = focal_length[None]

print("Start tracing...")
print("0 spp")

while True:

    if try_reset(t):
        if sample > 1:
            print ("\033[A                             \033[A")
            print("Camera moved, restart tracing...")
            print("0 spp")
        sample = 0
        linear_pixel.fill(0)
        bloom_pixel.fill(0)
        ray_count.fill(0)
        errors.fill(0)

    print ("\033[A                             \033[A")
    print(f"[{sample} spp]" )

    sample += 1
    trace()
    t += 1
    gui.set_image(pixel)
    gui.show()
    