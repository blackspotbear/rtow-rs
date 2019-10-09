extern crate image;
extern crate cgmath;
extern crate rand;
extern crate pbr;

use std::env;
use std::fs::File;
use std::f32;
use image::ColorType;
use image::png::PNGEncoder;
use cgmath::prelude::*;
use cgmath::Vector3;
use cgmath::vec3;
use rand::prelude::*;
use std::borrow::Borrow;

fn write_image(filename: &str, pixels: &[u8], bounds: (usize, usize)) -> Result<(), std::io::Error> {
    let output = File::create(filename)?;

    let encoder = PNGEncoder::new(output);

    encoder.encode(
        &pixels,
        bounds.0 as u32, bounds.1 as u32,
        ColorType::RGB(8)
    )?;

    Ok(())
}

struct Ray {
    a: Vector3<f32>,
    b: Vector3<f32>
}

impl Ray {
    fn new(a: Vector3<f32>, b: Vector3<f32>) -> Self {
        Ray { a: a, b: b }
    }
    fn origin(&self) -> Vector3<f32> { self.a }
    fn direction(&self) -> Vector3<f32> { self.b }
    fn point_at_parameter(&self, t: f32) -> Vector3<f32> { self.a + self.b * t }
}

//#[derive(Debug, Copy, Clone)]
#[derive(Copy, Clone)]
struct HitRecord<'a> {
    t: f32,
    p: Vector3<f32>,
    normal: Vector3<f32>,
    mat: &'a dyn Material
}

impl<'a> HitRecord<'a> { // impl<'a> defines 'a .
    #[allow(dead_code)]
    fn new(mat: &'a dyn Material) -> HitRecord<'a> {
        HitRecord {
            t: 0.0,
            p: vec3(0.0, 0.0, 0.0),
            normal: vec3(0.0, 0.0, 0.0),
            mat: mat
        }
    }
}

trait Hitable {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

struct Sphere {
    center: Vector3<f32>,
    radius: f32,
    mat: Box<dyn Material>
}

impl Sphere {
    fn new(center: Vector3<f32>, radius: f32, mat: Box<dyn Material>) -> Self {
        Sphere { center: center, radius: radius, mat: mat }
    }
}

impl Hitable for Sphere {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = r.origin() - self.center;
        let a = r.direction().dot(r.direction());
        let b = oc.dot(r.direction());
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - a * c;
        if discriminant > 0.0 {
            let temp = (-b - (b * b - a * c).sqrt()) / a;
            if temp < t_max && temp > t_min {
                let p = r.point_at_parameter(temp);
                return Some(HitRecord {
                    t: temp,
                    p: p,
                    normal: (p - self.center) / self.radius,
                    mat: self.mat.borrow()
                });
            }
            let temp = (-b + (b * b - a * c).sqrt()) / a;
            if temp < t_max && temp > t_min {
                let p = r.point_at_parameter(temp);
                return Some(HitRecord {
                    t: temp,
                    p: p,
                    normal: (p - self.center) / self.radius,
                    mat: self.mat.borrow()
                });
            }
        }
        None
    }
}

struct HitableList {
    hitable: Vec<Box<dyn Hitable>>
}

impl HitableList {
    fn new() -> Self {
        HitableList {
            hitable: Vec::new()
        }
    }
}

impl Hitable for HitableList {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut closest_so_far = t_max;
        let mut result = None;
        for i in 0..self.hitable.len() {
            if let Some(rec) = self.hitable[i].hit(r, t_min, closest_so_far) {
                closest_so_far = rec.t;
                result = Some(rec);
            }
        }
        result
    }
}

fn random_in_unit_sphere() -> Vector3<f32> {
    loop {
        let p = 2.0 * vec3(random::<f32>(), random::<f32>(), random::<f32>()) - vec3(1.0, 1.0, 1.0);
        if p.distance2(vec3(0.0, 0.0, 0.0)) < 1.0 {
            return p
        }
    }
}

fn random_in_unit_disk() -> Vector3<f32> {
    loop {
        let p = 2.0 * vec3(random::<f32>(), random::<f32>(), 0.0) - vec3(1.0, 1.0, 0.0);
        if p.dot(p) < 1.0 {
            return p
        }
    }
}

trait Material {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Ray, Vector3<f32>)>;
}

struct Lambertian {
    albedo: Vector3<f32>
}

impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, rec: &HitRecord) -> Option<(Ray, Vector3<f32>)> {
        let target = rec.p + rec.normal + random_in_unit_sphere();
        Some((Ray::new(rec.p, target - rec.p), self.albedo))
    }
}

fn reflect(v: Vector3<f32>, n: Vector3<f32>) -> Vector3<f32> {
    v - 2.0 * v.dot(n) * n
}

fn refract(v: Vector3<f32>, n: Vector3<f32>, ni_over_nt: f32) -> Option<Vector3<f32>> {
    let uv = v.normalize();
    let dt = uv.dot(n);
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
    if discriminant > 0.0 {
        Some(ni_over_nt * (uv - n * dt) - n * discriminant.sqrt())
    } else {
        None
    }
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0)
}

struct Metal {
    albedo: Vector3<f32>,
    fuzz: f32
}

impl Metal {
    fn new(albedo: Vector3<f32>, fuzz: f32) -> Self {
        Metal {
            albedo: albedo,
            fuzz: if fuzz < 1.0 { fuzz } else { 1.0 }
        }
    }
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Ray, Vector3<f32>)> {
        let reflected = reflect(r_in.direction().normalize(), rec.normal);
        let scattered = Ray::new(rec.p, reflected + self.fuzz * random_in_unit_sphere());
        if scattered.direction().dot(rec.normal) > 0.0 {
            Some((scattered, self.albedo))
        } else {
            None
        }
    }
}

struct Dielectric {
    ref_idx: f32
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Ray, Vector3<f32>)> {
        let outward_normal;
        let reflected = reflect(r_in.direction(), rec.normal);
        let ni_over_nt;
        let attenuation = vec3(1.0, 1.0, 1.0);
        let cosine;
        if r_in.direction().dot(rec.normal) > 0.0 {
            outward_normal = -rec.normal;
            ni_over_nt = self.ref_idx;
            cosine = self.ref_idx * r_in.direction().dot(rec.normal) / r_in.direction().distance(vec3(0.0, 0.0, 0.0));
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / self.ref_idx;
            cosine = -r_in.direction().dot(rec.normal) / r_in.direction().distance(vec3(0.0, 0.0, 0.0));
        }
        if let Some(refracted) = refract(r_in.direction(), outward_normal, ni_over_nt) {
            let reflect_prob = schlick(cosine, self.ref_idx);
            if rand::random::<f32>() > reflect_prob {
                return Some((Ray::new(rec.p, refracted), attenuation))
            }
        }
        Some((Ray::new(rec.p, reflected), attenuation))
    }
}

struct Camera {
    origin: Vector3<f32>,
    lower_left_corner: Vector3<f32>,
    horizontal: Vector3<f32>,
    vertical: Vector3<f32>,
    u: Vector3<f32>,
    v: Vector3<f32>,
    #[allow(dead_code)]
    w: Vector3<f32>,
    lens_radius: f32
}

impl Camera {
    fn new(lookfrom: Vector3<f32>, lookat: Vector3<f32>, vup: Vector3<f32>, vfov: f32, aspect: f32, aperture: f32, focus_dist: f32) -> Self {
        let lens_radius = aperture / 2.0;
        let theta = vfov * f32::consts::PI / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect * half_height;
        let origin = lookfrom;
        let w = (lookfrom - lookat).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);
        Camera {
            origin: origin,
            lower_left_corner: origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w,
            horizontal: 2.0 * half_width * focus_dist * u,
            vertical: 2.0 * half_height * focus_dist * v,
            u: u,
            v: v,
            w: w,
            lens_radius: lens_radius
        }
    }

    fn get_ray(&self, s: f32, t: f32) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset
        )
    }
}

fn color(r: &Ray, world: &dyn Hitable, depth: i32) -> Vector3<f32> {
    // Peter says:
    // Some of the reflected rays hit the object they are reflecting off of
    // not at exactly t=0, but instead at t=-0.0000001 or t=0.00000001 or
    // whatever floating point approximation the sphere intersector gives
    // us. So we need to ignore hits very near zero
    if let Some(rec) = world.hit(r, 0.0001, f32::MAX) {
        if depth < 50 {
            if let Some((scattered, attenuation)) = rec.mat.scatter(r, &rec) {
                return attenuation.mul_element_wise(color(&scattered, world, depth + 1));
            }
        }
        vec3(0.0, 0.0, 0.0)
    } else {
        let unit_direction = r.direction().normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0)
    }
}

fn random_scene() -> HitableList {
    let mut list = HitableList::new();

    list.hitable.push(Box::new(
        Sphere::new(vec3(0.0, -1000.0, 0.0), 1000.0,
        Box::new(Lambertian{albedo: vec3(0.5, 0.5, 0.5)}))
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = rand::random::<f32>();
            let center = vec3((a as f32) + 0.9 * rand::random::<f32>(), 0.2, (b as f32) + 0.9 * rand::random::<f32>());
            if (center - vec3(4.0, 0.2, 0.0)).distance(vec3(0.0, 0.0, 0.0)) > 0.9 {
                list.hitable.push(Box::new(Sphere::new(center, 0.2,
                    if choose_mat < 0.8 {
                        Box::new(Lambertian{albedo: vec3(
                            rand::random::<f32>() * rand::random::<f32>(),
                            rand::random::<f32>() * rand::random::<f32>(),
                            rand::random::<f32>() * rand::random::<f32>()
                        )})
                    } else if choose_mat < 0.95 {
                        Box::new(Metal::new(
                             vec3(
                                0.5 * (1.0 + rand::random::<f32>()),
                                0.5 * (1.0 + rand::random::<f32>()),
                                0.5 * (1.0 + rand::random::<f32>())
                            ),
                            0.5 * rand::random::<f32>()
                        ))
                    } else {
                        Box::new(Dielectric{ref_idx: 1.5})
                    }
                )));
            }
        }
    }

    list.hitable.push(Box::new(
        Sphere::new(
            vec3(0.0, 1.0, 0.0), 1.0,
            Box::new(Dielectric{ref_idx: 1.5})
        )
    ));
    list.hitable.push(Box::new(
        Sphere::new(
            vec3(-4.0, 1.0, 0.0), 1.0,
            Box::new(Lambertian{albedo: vec3(0.4, 0.2, 0.1)})
        )
    ));
    list.hitable.push(Box::new(
        Sphere::new(
            vec3(4.0, 1.0, 0.0), 1.0,
            Box::new(Metal{albedo: vec3(0.7, 0.6, 0.5), fuzz: 0.0})
        )
    ));

    list
}

fn sunrise_scene() -> HitableList {
    let mut list = HitableList::new();

    list.hitable.push(Box::new(Sphere::new(
        vec3(0.0, -1000.0, 0.0), 1000.0 - 0.01,
        Box::new(Lambertian{albedo: vec3(0.1, 0.8, 0.1)})
    )));

    list.hitable.push(Box::new(Sphere::new(
        vec3(0.0, -1000.0, 0.0), 1000.0,
        Box::new(Dielectric{ref_idx: 1.05})
    )));

    list.hitable.push(Box::new(Sphere::new(
        vec3(0.0, 0.0, -100.0), 10.0,
        Box::new(Lambertian{albedo: vec3(0.9, 0.1, 0.1)})
    )));

    list.hitable.push(Box::new(Sphere::new(
        vec3(0.0, 0.0, -100.0), 11.0,
        Box::new(Dielectric{ref_idx: 1.5})
    )));

    for (d, r) in [(14.0, 2.0), (16.0, 1.5), (17.5, 1.0)].iter() {
        for i in 0..8 {
            let th = 2.0 * f32::consts::PI / 8.0 * (i as f32);
            list.hitable.push(Box::new(Sphere::new(
                vec3(th.cos() * d, th.sin() * d, -100.0), *r,
                Box::new(Metal{albedo: vec3(1.0, 0.5, 0.5), fuzz: 0.0})
            )));
        }
    }

    for _ in 0..25 {
        let r = 1000.2;

        let th_range = f32::consts::PI * 0.001;
        let th = -th_range + rand::random::<f32>() * th_range * 2.0;
        let center = vec3(-th.sin() * r, th.cos() * r, 0.0);

        let th_range = f32::consts::PI * 0.005;
        let th = -rand::random::<f32>() * th_range;
        let center = vec3(center.x, th.cos() * center.y, th.sin() * center.y) - vec3(0.0, 1000.0, 0.0);

        let choose_mat = rand::random::<f32>();
        list.hitable.push(Box::new(Sphere::new(center, 0.2,
            if choose_mat < 0.8 {
                Box::new(Lambertian{albedo: vec3(
                    rand::random::<f32>() * rand::random::<f32>(),
                    rand::random::<f32>() * rand::random::<f32>(),
                    rand::random::<f32>() * rand::random::<f32>()
                )})
            } else if choose_mat < 0.95 {
                Box::new(Metal::new(
                        vec3(
                        0.5 * (1.0 + rand::random::<f32>()),
                        0.5 * (1.0 + rand::random::<f32>()),
                        0.5 * (1.0 + rand::random::<f32>())
                    ),
                    0.5 * rand::random::<f32>()
                ))
            } else {
                Box::new(Dielectric{ref_idx: 1.5})
            }
        )));
    }

    list
}

fn main() {
    let nx: usize = 640;
    let ny: usize = 360;
    let ns = 1000;

    let args: Vec<String> = env::args().collect();
    let (world, lookfrom, lookat, aperture) = if args.len() > 1 && args[1] == "sunrise" {
        (
            sunrise_scene(),
            vec3(0.0, 1.0, 3.0),
            vec3(0.0, 2.0, -100.0),
            0.01
        )
    } else {
        (
            random_scene(),
            vec3(12.0, 2.0, 3.0),
            vec3(0.0, 0.1, 0.0),
            0.1
        )
    };

    let dist_to_focus = (lookfrom - lookat).distance(vec3(0.0, 0.0, 0.0)) * 0.7;

    let cam = Camera::new(lookfrom, lookat, vec3(0.0, 1.0, 0.0), 20.0, (nx as f32) / (ny as f32), aperture, dist_to_focus);

    let mut pb = pbr::ProgressBar::new((nx * ny) as u64);
    pb.format("[=>-]");

    let mut buf = Vec::<u8>::new();
    for j in (0..ny).rev() {
        for i in 0..nx {
            let mut col = vec3(0.0, 0.0, 0.0);
            for _ in 0..ns {
                let u = ((i as f32) + rand::random::<f32>()) / (nx as f32);
                let v = ((j as f32) + rand::random::<f32>()) / (ny as f32);
                let r = cam.get_ray(u, v);
                col += color(&r, &world, 0);
            }
            col /= ns as f32;
            col = vec3(col[0].sqrt(), col[1].sqrt(), col[2].sqrt());
            buf.push((col.x * 255.0) as u8);
            buf.push((col.y * 255.0) as u8);
            buf.push((col.z * 255.0) as u8);
            pb.inc();
        }
    }

    pb.finish_print("done");

    write_image( "out.png", &buf, (nx, ny)).unwrap();
}
