// model.h -- header file
//

#include "vec3d.h"
#include <CL/opencl.h>
#include "ray-tracer.h"
#include <cassert>



typedef Vec3D<cl_float> Vector;

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
    size_t tri_count;
    TriangleBlock *child[2];
    Vector min, max;

public:
    TriangleBlock(const Vector &min_, const Vector &max_, Triangle **ptr, size_t count) :
        tri(ptr), tri_count(count), min(min_), max(max_)
    {
        child[0] = child[1] = 0;
    }

    ~TriangleBlock()
    {
        delete child[0];  delete child[1];
    }

    size_t subdivide(size_t threshold);  // returns block count
    size_t count_points();

    void fill_data(Group *&group, AABB *&aabb,
        Vertex *vtx_buf, cl_uint &vtx_pos, cl_uint *tri_buf, cl_uint &tri_pos);
};


class Model
{
    ModelVertex *vtx;
    Triangle *tri, **tri_ptr;
    size_t vtx_count, tri_count;
    TriangleBlock *root;

    void prepare();

public:
    Model() : vtx(0), tri(0), vtx_count(0), tri_count(0), root(0)
    {
    }

    ~Model()
    {
        delete [] vtx;  delete [] tri;  delete [] tri_ptr;  delete root;
    }

    size_t load(const char *file);  // returns triangle count

    size_t subdivide(size_t threshold)  // returns block count
    {
        return root->subdivide(threshold);
    }

    size_t count_points()
    {
        return root->count_points();
    }

    void fill_data(Group *group, AABB *aabb, Vertex *vtx_buf, cl_uint *tri_buf)
    {
        cl_uint vtx_pos = 0, tri_pos = 0;
        return root->fill_data(group, aabb, vtx_buf, vtx_pos, tri_buf, tri_pos);
    }
};

