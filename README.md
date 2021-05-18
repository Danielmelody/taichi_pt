# Taichi Path tracer

![](result.png)

A reatime path tracer written in [taichi](https://github.com/taichi-dev/taichi)

#### Features:
* Global illumination via unbiased Monte Carlo path tracing
* Physically based Specular shading(GGX)
* Lambert diffuse shading
* Ray-Sphere intersection
* Unbiasd russain roule 
* Antialiasing via super-sampling
* Multiple Importance Sampling
  * Balance heuristic 
  * Cosine-weighted pdf
  * ggx normal weighted pdf 
* Gamma correction of final result

The 5 balls in the scene are:
 1. rough golden ball
 2. smooth ceramics ball
 3. the light source ball
 4. huge ground rough iron ball
 5. smooth metal ball
#### Usage

```bash
pip3 install taichi
python3 pt.py
```

and you a ready to go.
To see a interactive result **zoom in/out, press and drag the mouse to see different rendering result from different camera position**
