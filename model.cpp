// model.cpp -- model loading
//

#include "model.h"
#include <cstdio>

using namespace std;



void set_camera(Camera &cam, size_t width, size_t height, cl_float tan_fov,
    const Vector &pos, const Vector &view, const Vector &up)
{
    cl_float scale = tan_fov / sqrt(width * cl_float(width) + height * cl_float(height));
    Vector dir = normalize(view), dx = scale * normalize(view % up), dy = dx % dir;

    cam.eye = to_float3(pos);
    cam.top_left = to_float3(dir - (width * dx + height * dy) / 2);
    cam.dx = to_float3(dx);  cam.dy = to_float3(dy);
    cam.width = width;  cam.height = height;
}



size_t TriangleBlock::subdivide(size_t tri_threshold, size_t aabb_threshold, bool root)
{
    assert(!child[0] && !child[1]);
    if(tri_count < tri_threshold)return 1;

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

    size_t block_count =
        child[0]->subdivide(tri_threshold, aabb_threshold, false) +
        child[1]->subdivide(tri_threshold, aabb_threshold, false);
    if(!root && block_count < aabb_threshold)return block_count;
    aabb_count = block_count;  return 1;
}

void TriangleBlock::reserve(ResourceManager &mngr)
{
    if(child[0])
    {
        child[0]->reserve(mngr);  child[1]->reserve(mngr);  if(!aabb_count)return;
        mngr.reserve_groups(1);  mngr.reserve_aabbs(aabb_count);  return;
    }

    mngr.reserve_groups(1);  mngr.reserve_triangles(tri_count);  int pos = 0;
    for(size_t i = 0; i < tri_count; i++)
    {
        if(tri[i]->pt[0]->index < 0)tri[i]->pt[0]->index = pos++;
        if(tri[i]->pt[1]->index < 0)tri[i]->pt[1]->index = pos++;
        if(tri[i]->pt[2]->index < 0)tri[i]->pt[2]->index = pos++;
    }
    for(size_t i = 0; i < tri_count; i++)
        tri[i]->pt[0]->index = tri[i]->pt[1]->index = tri[i]->pt[2]->index = -1;
    assert(pos < (1 << 10));  mngr.reserve_vertices(vtx_count = pos);
}

inline cl_uint put_vertex(ModelVertex *vtx, Vector &min, Vector &max, Vertex *buf, int &pos)
{
    int index = vtx->index;  if(index >= 0)return index;  index = vtx->index = pos++;
    buf[index].pos = to_float3(vtx->pos);  buf[index].norm = to_float3(vtx->norm);
    update_bounds(min, max, vtx->pos);  return index;
}

cl_uint TriangleBlock::fill(ResourceManager &mngr, cl_uint material_id, cl_uint *aabb_index)
{
    if(child[0])
    {
        if(aabb_count)
        {
            size_t grp_pos = mngr.get_groups(1);  Group *grp = mngr.group(grp_pos);
            cl_uint aabb_sub = grp->aabb.aabb_offs = mngr.get_aabbs(aabb_count);
            grp->aabb.aabb_count = aabb_count;  grp->aabb.flags = 0;

            child[0]->fill(mngr, material_id, &aabb_sub);
            child[1]->fill(mngr, material_id, &aabb_sub);
            assert(aabb_sub == grp->aabb.aabb_offs + aabb_count);
            min = vec_min(child[0]->min, child[1]->min);
            max = vec_max(child[0]->max, child[1]->max);

            cl_uint group_id = make_group_id(grp_pos, tr_ortho, sh_aabb);
            if(aabb_index)
            {
                AABB *aabb = mngr.aabb((*aabb_index)++);
                aabb->min = to_float3(min);  aabb->max = to_float3(max);
                aabb->group_id = group_id;  aabb->local_id = 0;
            }
            return group_id;
        }
        else
        {
            child[0]->fill(mngr, material_id, aabb_index);
            child[1]->fill(mngr, material_id, aabb_index);
            min = vec_min(child[0]->min, child[1]->min);
            max = vec_max(child[0]->max, child[1]->max);
        }
        return 0;
    }

    size_t grp_pos = mngr.get_groups(1);  Group *grp = mngr.group(grp_pos);
    Vertex *vtx_buf = mngr.vertex(grp->mesh.vtx_offs = mngr.get_vertices(vtx_count));
    cl_uint *tri_buf = mngr.triangle(grp->mesh.tri_offs = mngr.get_triangles(tri_count));
    grp->mesh.tri_count = tri_count;  grp->mesh.material_id = material_id;

    int pos = 0;  init_bounds(min, max);
    for(size_t i = 0; i < tri_count; i++)
    {
        cl_uint index0 = put_vertex(tri[i]->pt[0], min, max, vtx_buf, pos);
        cl_uint index1 = put_vertex(tri[i]->pt[1], min, max, vtx_buf, pos);
        cl_uint index2 = put_vertex(tri[i]->pt[2], min, max, vtx_buf, pos);
        tri_buf[i] = index0 | index1 << 10 | index2 << 20;
    }
    for(size_t i = 0; i < tri_count; i++)
        tri[i]->pt[0]->index = tri[i]->pt[1]->index = tri[i]->pt[2]->index = -1;
    assert(size_t(pos) == vtx_count);

    cl_uint group_id = make_group_id(grp_pos, tr_ortho, sh_mesh);
    if(aabb_index)
    {
        AABB *aabb = mngr.aabb((*aabb_index)++);
        aabb->min = to_float3(min);  aabb->max = to_float3(max);
        aabb->group_id = group_id;  aabb->local_id = 0;
    }
    return group_id;
}


bool Model::load(const char *file)
{
    assert(!vtx && !tri);  FILE *input = fopen(file, "r");  if(!input)return false;

    char buf[256], fmt[] = "%f %f %f %*f %*f %*f %*f ";

    static const char *header1 =
        "ply "
        "format ascii 1.0 "
        "comment %*[^\n] "
        "element vertex %zu "
        "property float x "
        "property float y "
        "property float z ";

    static const char *header2 =
        "property float %255s ";

    static const char *header3 =
        "element face %zu "
        "property list uchar %*[ui]nt vertex_indices "
        "end_header ";

    if(fscanf(input, header1, &vtx_count) != 1 || !vtx_count)
    {
        fclose(input);  return false;
    }
    size_t len = 9;
    while(fscanf(input, header2, buf) == 1)len += 4;
    if(len >= sizeof(fmt))
    {
        fclose(input);  return false;
    }
    fmt[len] = '\0';
    if(fscanf(input, header3, &tri_count) != 1 || !tri_count)
    {
        fclose(input);  return false;
    }
    vtx = new ModelVertex[vtx_count];  tri = new Triangle[tri_count];  tri_ptr = new Triangle *[tri_count];
    for(size_t i = 0; i < vtx_count; i++)
    {
        if(fscanf(input, fmt, &vtx[i].pos.x, &vtx[i].pos.y, &vtx[i].pos.z) != 3)
        {
            fclose(input);  return false;
        }
    }
    for(size_t i = 0, index[3]; i < tri_count; i++)
    {
        if(fscanf(input, "3 %zu %zu %zu ", &index[0], &index[1], &index[2]) != 3 ||
            index[0] >= vtx_count || index[1] >= vtx_count || index[2] >= vtx_count)
        {
            fclose(input);  return false;
        }
        tri[i].pt[0] = &vtx[index[0]];  tri[i].pt[1] = &vtx[index[1]];  tri[i].pt[2] = &vtx[index[2]];
    }
    fclose(input);  prepare();  return true;
}

void Model::prepare()
{
    assert(!root);
    for(size_t i = 0; i < vtx_count; i++)
    {
        vtx[i].norm = Vector(0, 0, 0);  vtx[i].index = -1;
    }
    Vector min, max;  init_bounds(min, max);
    for(size_t i = 0; i < tri_count; i++)
    {
        Vector pt[3] = {tri[i].pt[0]->pos, tri[i].pt[1]->pos, tri[i].pt[2]->pos};
        tri[i].center = (pt[0] + pt[1] + pt[2]) / 3;  Vector norm = (pt[1] - pt[0]) % (pt[2] - pt[0]);
        tri[i].pt[0]->norm += norm;  tri[i].pt[1]->norm += norm;  tri[i].pt[2]->norm += norm;
        update_bounds(min, max, tri[i].center);  tri_ptr[i] = &tri[i];
    }
    for(size_t i = 0; i < vtx_count; i++)vtx[i].norm /= vtx[i].norm.len();
    root = new TriangleBlock(min, max, tri_ptr, tri_count);
}

void Model::put(AABB &aabb, const Matrix &mat, cl_uint local_id)
{
    assert(group_id);  Vector min, max;  init_bounds(min, max);
    for(size_t i = 0; i < vtx_count; i++)update_bounds(min, max, mat * vtx[i].pos);
    aabb.min = to_float3(min);  aabb.max = to_float3(max);
    aabb.group_id = group_id;  aabb.local_id = local_id;
}
