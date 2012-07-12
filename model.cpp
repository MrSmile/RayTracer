// model.cpp -- model loading
//

#include "model.h"
#include <algorithm>
#include <limits>
#include <cstdio>

using namespace std;



size_t TriangleBlock::subdivide(size_t threshold)
{
    assert(!child[0] && !child[1]);
    if(tri_count < threshold)return 1;

    Vector delta = max - min;  cl_float Vector::*axis;
    if(delta.x > delta.y && delta.x > delta.z)axis = &Vector::x;
    else if(delta.y > delta.z)axis = &Vector::y;
    else axis = &Vector::z;

    size_t center = tri_count / 2;
    sort(tri, tri + tri_count, TriangleCompare(axis));
    child[0] = new TriangleBlock(min, max, tri, center);
    child[1] = new TriangleBlock(min, max, tri + center, tri_count - center);
    child[0]->max.*axis = tri[center - 1]->center.*axis;
    child[1]->min.*axis = tri[center]->center.*axis;

    return child[0]->subdivide(threshold) + child[1]->subdivide(threshold);
}

size_t TriangleBlock::count_points()
{
    if(child[0])return child[0]->count_points() + child[1]->count_points();

    int pos = 0;
    for(size_t i = 0; i < tri_count; i++)
    {
        if(tri[i]->pt[0]->index < 0)tri[i]->pt[0]->index = pos++;
        if(tri[i]->pt[1]->index < 0)tri[i]->pt[1]->index = pos++;
        if(tri[i]->pt[2]->index < 0)tri[i]->pt[2]->index = pos++;
    }
    for(size_t i = 0; i < tri_count; i++)
        tri[i]->pt[0]->index = tri[i]->pt[1]->index = tri[i]->pt[2]->index = -1;
    assert(pos < (1 << 10));  return pos;
}

inline cl_float3 to_float3(const Vector &vec)
{
    cl_float3 res;  res.s[0] = vec.x;  res.s[1] = vec.y;  res.s[2] = vec.z;  return res;
}

inline void update_limits(Vector &min, Vector &max, const Vector &vec)
{
    if(min.x > vec.x)min.x = vec.x;  if(min.y > vec.y)min.y = vec.y;  if(min.z > vec.z)min.z = vec.z;
    if(max.x < vec.x)max.x = vec.x;  if(max.y < vec.y)max.y = vec.y;  if(max.z < vec.z)max.z = vec.z;
}

inline cl_uint put_vertex(ModelVertex *vtx, Vector &min, Vector &max, Vertex *buf, int &pos)
{
    int index = vtx->index;  if(index >= 0)return index;  index = vtx->index = pos++;
    buf[index].pos = to_float3(vtx->pos);  buf[index].norm = to_float3(vtx->norm);
    update_limits(min, max, vtx->pos);  return index;
}

void TriangleBlock::fill_data(Group *&group, AABB *&aabb,
    Vertex *vtx_buf, cl_uint &vtx_pos, cl_uint *tri_buf, cl_uint &tri_pos)
{
    if(child[0])
    {
        child[0]->fill_data(group, aabb, vtx_buf, vtx_pos, tri_buf, tri_pos);
        child[1]->fill_data(group, aabb, vtx_buf, vtx_pos, tri_buf, tri_pos);
        return;
    }

    int pos = 0;  Vector min, max;
    vtx_buf += vtx_pos;  tri_buf += tri_pos;
    min.x = min.y = min.z = numeric_limits<cl_float>::infinity();
    max.x = max.y = max.z = -numeric_limits<cl_float>::infinity();
    for(size_t i = 0; i < tri_count; i++)
    {
        cl_uint index0 = put_vertex(tri[i]->pt[0], min, max, vtx_buf, pos);
        cl_uint index1 = put_vertex(tri[i]->pt[1], min, max, vtx_buf, pos);
        cl_uint index2 = put_vertex(tri[i]->pt[2], min, max, vtx_buf, pos);
        tri_buf[i] = index0 | index1 << 10 | index2 << 20;
    }
    for(size_t i = 0; i < tri_count; i++)
        tri[i]->pt[0]->index = tri[i]->pt[1]->index = tri[i]->pt[2]->index = -1;

    group->mesh.vtx_offs = vtx_pos;  vtx_pos += pos;
    group->mesh.tri_offs = tri_pos;  tri_pos += tri_count;
    group->mesh.tri_count = tri_count;  group++;

    aabb->min = to_float3(min);  aabb->max = to_float3(max);  aabb++;
}


size_t Model::load(const char *file)
{
    assert(!vtx && !tri);  FILE *input = fopen(file, "r");  if(!input)return 0;

    static const char *header =
        "ply "
        "format ascii 1.0 "
        "comment zipper output "
        "element vertex %zu "
        "property float x "
        "property float y "
        "property float z "
        "property float confidence "
        "property float intensity "
        "element face %zu "
        "property list uchar int vertex_indices "
        "end_header ";

    if(fscanf(input, header, &vtx_count, &tri_count) != 2 || !vtx_count || !tri_count)
    {
        fclose(input);  return 0;
    }
    vtx = new ModelVertex[vtx_count];  tri = new Triangle[tri_count];  tri_ptr = new Triangle *[tri_count];
    for(size_t i = 0; i < vtx_count; i++)
    {
        if(fscanf(input, "%f %f %f %*f %*f ", &vtx[i].pos.x, &vtx[i].pos.y, &vtx[i].pos.z) != 3)
        {
            fclose(input);  return 0;
        }
    }
    for(size_t i = 0, index[3]; i < tri_count; i++)
    {
        if(fscanf(input, "3 %zu %zu %zu ", &index[0], &index[1], &index[2]) != 3 ||
            index[0] >= vtx_count || index[1] >= vtx_count || index[2] >= vtx_count)
        {
            fclose(input);  return 0;
        }
        tri[i].pt[0] = &vtx[index[0]];  tri[i].pt[1] = &vtx[index[1]];  tri[i].pt[2] = &vtx[index[2]];
    }
    fclose(input);  prepare();  return tri_count;
}

void Model::prepare()
{
    assert(!root);  Vector min, max;
    min.x = min.y = min.z = numeric_limits<cl_float>::infinity();
    max.x = max.y = max.z = -numeric_limits<cl_float>::infinity();
    for(size_t i = 0; i < vtx_count; i++)
    {
        vtx[i].norm = Vector(0, 0, 0);  vtx[i].index = -1;
    }
    for(size_t i = 0; i < tri_count; i++)
    {
        Vector pt[3] = {tri[i].pt[0]->pos, tri[i].pt[1]->pos, tri[i].pt[2]->pos};
        tri[i].center = (pt[0] + pt[1] + pt[2]) / 3;  Vector norm = (pt[1] - pt[0]) ^ (pt[2] - pt[0]);
        tri[i].pt[0]->norm += norm;  tri[i].pt[1]->norm += norm;  tri[i].pt[2]->norm += norm;
        update_limits(min, max, tri[i].center);  tri_ptr[i] = &tri[i];
    }
    for(size_t i = 0; i < vtx_count; i++)vtx[i].norm /= vtx[i].norm.len();
    root = new TriangleBlock(min, max, tri_ptr, tri_count);
}
