from locale import currency
from posix import F_OK
from PIL.Image import BILINEAR
from _pytest.store import D
from astroid.interpreter._import import spec
import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu, excepthook=True)

inf = 1e10
eps = 1e-3


class Image:
    def __init__(self, path):
        self.img = ti.imread(path)
        tex_w, tex_h = self.img.shape[0:2]
        self.field = ti.Vector.field(3, dtype=ti.u8, shape=(tex_w, tex_h))

    def load(self):
        self.field.from_numpy(self.img)


@ti.func
def tex2d(tex_field, uv):
    tex_w, tex_h = tex_field.shape[0:2]
    p = uv * ti.Vector([tex_w - 1, tex_h - 1])
    l = ti.floor(p)
    t = p - l

    # bilinear interp
    return (
        ((tex_field[p] * (1 - t[0])) +
         tex_field[p + ti.Vector([1, 0])] * t[0]) * (1 - t[1])
        + (tex_field[p + ti.Vector([1, 0])] * (1 - t[0]) + tex_field[p + ti.Vector([1, 1])] * t[0]) * t[1])


@ti.data_oriented
class Spheres:
    def from_numpy(self, center_radius, albedos, emissions, roughness, metallics, iors):
        self.center_radius = ti.Vector.field(
            4, dtype=ti.f32, shape=center_radius.shape[:1])
        self.albedos = ti.Vector.field(
            3, dtype=ti.f32, shape=albedos.shape[:1])
        self.emissions = ti.Vector.field(
            3, dtype=ti.f32, shape=emissions.shape[:1])
        self.roughness = ti.field(
            dtype=ti.f32, shape=roughness.shape[:1])
        self.metallics = ti.field(
            dtype=ti.f32, shape=metallics.shape[:1])
        self.iors = ti.field(
            dtype=ti.f32, shape=iors.shape[:1])

        self.center_radius.from_numpy(center_radius)
        self.albedos.from_numpy(albedos)
        self.emissions.from_numpy(emissions)
        self.roughness.from_numpy(roughness)
        self.metallics.from_numpy(metallics)
        self.iors.from_numpy(iors)

    @ ti.func
    def intersect(self, o, d):
        min_t = inf
        min_index = -1
        sp_index = 0
        for i in ti.ndrange(5):
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

linear_pixel = ti.Vector.field(3, dtype=float, shape=(width, height))
pixel = ti.Vector.field(3, dtype=float, shape=(width, height))
spheres = Spheres()
skybox = Image('skybox.jpg')  # z > 0


last_camera_pos = ti.field(ti.f32, 3)
camera_pos = ti.field(ti.f32, 3)

spheres.from_numpy(
    center_radius=np.array([
        [-2., 0.0, -3., 2.5],
        [1.0, -1.5, 1.0, 1.],
        [0.0, 2., 0., 0.5],
        [0.0, -500., 0., 497.5],
        [-1, -2.0, 0.5, 0.5]]).astype(np.float32),
    albedos=np.array([[1.0, 0.8, 0.2],
                      [1.0, 1.0, 1.0],
                      [0.0, 0.0, 0.0],
                      [1.0, 0.8, 0.6],
                      [1.0, 1.0, 1.0]]).astype(np.float32),
    emissions=np.array([[0, 0, 0], [0, 0, 0], [15, 15, 15], [
                       0, 0, 0], [0, 0, 0]]).astype(np.float32),
    roughness=np.array([0.3, 0.0, 0.0, 0.5, 0.0]).astype(np.float32),
    metallics=np.array([1.0, 0.0, 0.8, 0.9, 1.0]).astype(np.float32),
    iors=np.array([0.495, 1.4, 2.0, 2.90, 2.5]).astype(np.float32),
)

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
            ti.Vector([dir.x, dir.y]) / dir.z / 8
    if dir.z <= 0 and -dir.z >= abs(dir.y) - eps and -dir.z >= abs(dir.x) - eps:
        coor = ti.Vector([7 / 8, 1 / 2]) + \
            ti.Vector([-dir.x, dir.y]) / -dir.z / 8
    if dir.x <= 0 and -dir.x >= abs(dir.y) - eps and -dir.x >= abs(dir.z) - eps:
        coor = ti.Vector([1 / 8, 1 / 2]) + \
            ti.Vector([dir.z, dir.y]) / -dir.x / 8
    if dir.x >= 0 and dir.x >= abs(dir.y) - eps and dir.x >= abs(dir.z) - eps:
        coor = ti.Vector([5 / 8, 1 / 2]) + \
            ti.Vector([-dir.z, dir.y]) / dir.x / 8
    if dir.y >= 0 and dir.y >= abs(dir.x) - eps and dir.y >= abs(dir.z) - eps:
        coor = ti.Vector([3 / 8, 5 / 6]) + \
            ti.Vector([dir.x, -dir.z]) / dir.y / 8
    if dir.y <= 0 and -dir.y >= abs(dir.x) - eps and -dir.y >= abs(dir.z) - eps:
        coor = ti.Vector([3 / 8, 1 / 6]) + \
            ti.Vector([dir.x, dir.z]) / -dir.y / 8
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


@ ti.func
def halton(b, i):
    r = 0.0
    f = 1.0
    while i > 0:
        f = f / float(b)
        r = r + f * float(i % b)
        i = int(ti.floor(float(i) / float(b)))
    return r


gui = ti.GUI("Path Tracer", res=(width, height))


@ ti.kernel
def trace(sample: ti.u32):
    for i, j in linear_pixel:
        o = ti.Vector([camera_pos[0], camera_pos[1], camera_pos[2]])
        forward = -o.normalized()
        u = ti.Vector([0.0, 1.0, 0.0]).cross(forward).normalized()
        v = forward.cross(u).normalized()
        u = -u
        # anti aliasing
        d = (i + ti.random() - width / 2.0) / width * u * 1.5 \
            + (j + ti.random() - height / 2.0) / width * v * 1.5 \
            + width / width * forward

        d = d.normalized()
        uv = cubemap_coord(d)
        albedo_factor = ti.Vector([1.0, 1.0, 1.0])
        radiance = ti.Vector([0.0, 0.0, 0.0])
        for step in ti.ndrange(32):
            sp = spheres.intersect(o, d)
            uv = cubemap_coord(d)
            if sp[1] > -1:
                sp_index = int(sp[1])
                p = sp[0] * d + o
                c_ = spheres.center_radius[sp_index]
                c = ti.Vector([c_[0], c_[1], c_[2]])
                n = (p - c).normalized()
                wo = -d
                albedo = spheres.albedos[sp_index]
                metallic = spheres.metallics[sp_index]
                ior = spheres.iors[sp_index]
                roughness = ti.max(0.04, spheres.roughness[sp_index])
                f0 = (1.0 - ior) / (1.0 + ior)
                f0 = f0 * f0
                f0 = lerp(f0, luma(albedo), metallic)
                wi = reflect(-wo, n)
                radiance += spheres.emissions[sp_index] * albedo_factor

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
                radiance += albedo_factor * tex2d(skybox.field, uv) / 255
                break

        linear_pixel[i, j] = (linear_pixel[i, j] *
                              (sample - 1) + radiance) / sample
        pixel[i, j] = ti.sqrt(linear_pixel[i, j])


start_cursor = [0.0, 0.0]


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
            for i in range(3):
                camera_pos[i] *= (1.0 + dt)
            return True

    return False


sample = 0
t = 1
last_camera_pos[2] = 10.0
camera_pos[2] = 10.0

while True:
    if sample % 64 == 0:
        print(sample, " spp")

    if try_reset(t):
        sample = 0
        linear_pixel.fill(0)

    sample += 1
    trace(sample)
    t += 1
    gui.set_image(linear_pixel)
    gui.show()
