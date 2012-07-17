// model.h -- header file
//

#include "vec3d.h"
#include <CL/opencl.h>
#include "ray-tracer.h"
#include <algorithm>
#include <cassert>
#include <limits>



inline cl_uint make_group_id(size_t index, int transform, int shader)
{
    return index | transform << GROUP_TR_SHIFT | shader << GROUP_SH_SHIFT;
}


class ResourceManager
{
    Group *grp_;  AABB *aabb_;  Vertex *vtx_;  cl_uint *tri_;
    size_t grp_count_, aabb_count_, vtx_count_, tri_count_;
    cl_uint grp_pos_, aabb_pos_, vtx_pos_, tri_pos_;


public:
    ResourceManager() : grp_(0), aabb_(0), vtx_(0), tri_(0),
        grp_count_(0), aabb_count_(0), vtx_count_(0), tri_count_(0),
        grp_pos_(0), aabb_pos_(0), vtx_pos_(0), tri_pos_(0)
    {
    }

    ~ResourceManager()
    {
        delete [] grp_;  delete [] aabb_;  delete [] vtx_;  delete [] tri_;
    }

    void alloc()
    {
        assert(!grp_ && !aabb_ && !vtx_ && !tri_);
        if(grp_count_)grp_ = new Group[grp_count_];
        if(aabb_count_)aabb_ = new AABB[aabb_count_];
        if(vtx_count_)vtx_ = new Vertex[vtx_count_];
        if(tri_count_)tri_ = new cl_uint[tri_count_];
    }

    bool full() const
    {
        return grp_pos_ == grp_count_ && aabb_pos_ == aabb_count_ &&
            vtx_pos_ == vtx_count_ && tri_pos_ == tri_count_;
    }


    Group *group(size_t index)
    {
        assert(grp_ && index < grp_pos_);  return grp_ + index;
    }

    AABB *aabb(size_t index)
    {
        assert(aabb_ && index < aabb_pos_);  return aabb_ + index;
    }

    Vertex *vertex(size_t index)
    {
        assert(vtx_ && index < vtx_pos_);  return vtx_ + index;
    }

    cl_uint *triangle(size_t index)
    {
        assert(tri_ && index < tri_pos_);  return tri_ + index;
    }


    size_t group_count() const
    {
        return grp_count_;
    }

    size_t aabb_count() const
    {
        return aabb_count_;
    }

    size_t vertex_count() const
    {
        return vtx_count_;
    }

    size_t triangle_count() const
    {
        return tri_count_;
    }


    void reserve_groups(size_t n)
    {
        assert(!grp_);  grp_count_ += n;
    }

    void reserve_aabbs(size_t n)
    {
        assert(!aabb_);  aabb_count_ += n;
    }

    void reserve_vertices(size_t n)
    {
        assert(!vtx_);  vtx_count_ += n;
    }

    void reserve_triangles(size_t n)
    {
        assert(!tri_);  tri_count_ += n;
    }


    cl_uint get_groups(size_t n)
    {
        assert(!n || grp_ && grp_pos_ + n <= grp_count_);
        cl_uint res = grp_pos_;  grp_pos_ += n;  return res;
    }

    cl_uint get_aabbs(size_t n)
    {
        assert(!n || aabb_ && aabb_pos_ + n <= aabb_count_);
        cl_uint res = aabb_pos_;  aabb_pos_ += n;  return res;
    }

    cl_uint get_vertices(size_t n)
    {
        assert(!n || vtx_ && vtx_pos_ + n <= vtx_count_);
        cl_uint res = vtx_pos_;  vtx_pos_ += n;  return res;
    }

    cl_uint get_triangles(size_t n)
    {
        assert(!n || tri_ && tri_pos_ + n <= tri_count_);
        cl_uint res = tri_pos_;  tri_pos_ += n;  return res;
    }
};



typedef Vec3D<cl_float> Vector;

inline void init_bounds(Vector &min, Vector &max)
{
    min.x = min.y = min.z = +std::numeric_limits<cl_float>::infinity();
    max.x = max.y = max.z = -std::numeric_limits<cl_float>::infinity();
}

inline Vector vec_min(const Vector &vec1, const Vector &vec2)
{
    cl_float x = std::min(vec1.x, vec2.x);
    cl_float y = std::min(vec1.y, vec2.y);
    cl_float z = std::min(vec1.z, vec2.z);
    return Vector(x, y, z);
}

inline Vector vec_max(const Vector &vec1, const Vector &vec2)
{
    cl_float x = std::max(vec1.x, vec2.x);
    cl_float y = std::max(vec1.y, vec2.y);
    cl_float z = std::max(vec1.z, vec2.z);
    return Vector(x, y, z);
}

inline void update_bounds(Vector &min, Vector &max, const Vector &vec)
{
    min = vec_min(min, vec);  max = vec_max(max, vec);
}

inline Vector operator * (const Matrix &mat, const Vector &vec)
{
    cl_float x = mat.x.s[0] * vec.x + mat.x.s[1] * vec.y + mat.x.s[2] * vec.z + mat.x.s[3];
    cl_float y = mat.y.s[0] * vec.x + mat.y.s[1] * vec.y + mat.y.s[2] * vec.z + mat.y.s[3];
    cl_float z = mat.z.s[0] * vec.x + mat.z.s[1] * vec.y + mat.z.s[2] * vec.z + mat.z.s[3];
    return Vector(x, y, z);
}

inline cl_float3 to_float3(const Vector &vec)
{
    cl_float3 res;  res.s[0] = vec.x;  res.s[1] = vec.y;  res.s[2] = vec.z;  return res;
}


struct ModelVertex
{
    Vector pos, norm;
    int index;
};

struct Triangle
{
    Vector center;
    ModelVertex *pt[3];
};

struct TriangleCompare
{
    cl_float Vector::*axis;

    TriangleCompare(cl_float Vector::*axis_) : axis(axis_)
    {
    }

    bool operator () (const Triangle *tri1, const Triangle *tri2)
    {
        return tri1->center.*axis < tri2->center.*axis;
    }
};


class TriangleBlock
{
    Triangle **tri;
    size_t aabb_count, vtx_count, tri_count;
    TriangleBlock *child[2];
    Vector min, max;


public:
    TriangleBlock(const Vector &min_, const Vector &max_, Triangle **ptr, size_t count) :
        tri(ptr), aabb_count(0), vtx_count(0), tri_count(count), min(min_), max(max_)
    {
        child[0] = child[1] = 0;
    }

    ~TriangleBlock()
    {
        delete child[0];  delete child[1];
    }


    size_t subdivide(size_t tri_threshold, size_t aabb_threshold, bool root = true);  // returns aabb_count

    void reserve(ResourceManager &mngr);
    cl_uint fill(ResourceManager &mngr, cl_uint material_id, cl_uint *aabb_index = 0);  // returns group_id
};


class Model
{
    ModelVertex *vtx;
    Triangle *tri, **tri_ptr;
    size_t vtx_count, tri_count;
    TriangleBlock *root;
    cl_uint group_id;


    void prepare();


public:
    Model() : vtx(0), tri(0), vtx_count(0), tri_count(0), root(0), group_id(0)
    {
    }

    ~Model()
    {
        delete [] vtx;  delete [] tri;  delete [] tri_ptr;  delete root;
    }


    bool load(const char *file);

    void subdivide(size_t tri_threshold, size_t aabb_threshold)
    {
        root->subdivide(tri_threshold, aabb_threshold);
    }

    void reserve(ResourceManager &mngr)
    {
        root->reserve(mngr);
    }

    void fill(ResourceManager &mngr, cl_uint material_id)
    {
        group_id = root->fill(mngr, material_id);
    }

    void put(AABB &aabb, const Matrix &mat, cl_uint local_id);
};

