// sort.cl -- sorting algorithm
//


uint2 local_scan(local uint *buf, uint val)  // buf[2 * UNIT_WIDTH]
{
    const uint index = get_local_id(0) + UNIT_WIDTH;  uint res = val;
    for(uint offs = 1; offs < UNIT_WIDTH; offs *= 2)
    {
        buf[index] = res;  barrier(CLK_LOCAL_MEM_FENCE);
        res += buf[index - offs];  barrier(CLK_LOCAL_MEM_FENCE);
    }
    buf[index] = res;  barrier(CLK_LOCAL_MEM_FENCE);
    return (uint2)(res - val, buf[2 * UNIT_WIDTH - 1]);
}

void KERNEL local_count(const global uint2 *val, global uint *local_index, global uint *global_index, uint order)
{
    const uint group = get_group_id(0);  val += group * SORT_WIDTH;
    local_index += group * SORT_WIDTH;  global_index += group * RADIX_MAX;
    const uint index = get_local_id(0);  uint2 data[SORT_BLOCK];
    for(uint i = 0; i < SORT_BLOCK; i++)data[i] = val[index * SORT_BLOCK + i];

    uint count[RADIX_MAX];
    for(uint i = 0; i < RADIX_MAX; i++)count[i] = 0;

    uint pos[SORT_BLOCK];  order *= RADIX_SHIFT;
    for(uint i = 0; i < SORT_BLOCK; i++)pos[i] = count[(data[i].s0 >> order) & RADIX_MASK]++;

    local uint buf[2 * UNIT_WIDTH];  buf[index] = 0;
    for(uint i = 0; i < RADIX_MAX; i++)
    {
        uint2 res = local_scan(buf, count[i]);  count[i] = res.s0;
        if(!index)global_index[i] = res.s1;  barrier(CLK_LOCAL_MEM_FENCE);
    }
    for(uint i = 0; i < SORT_BLOCK; i++)
        local_index[index * SORT_BLOCK + i] = pos[i] + count[(data[i].s0 >> order) & RADIX_MASK];
}

void KERNEL global_count(global uint *global_index, uint group_count)  // single unit
{
    const uint index = get_global_id(0);

    uint count[RADIX_MAX];
    for(uint i = 0; i < RADIX_MAX; i++)
    {
        count[i] = 0;
        for(uint pos = index; pos < group_count; pos += UNIT_WIDTH)
        {
            uint k = pos * RADIX_MAX + i;  uint n = global_index[k];
            global_index[k] = count[i];  count[i] += n;
        }
    }

    local uint buf[2 * UNIT_WIDTH];  buf[index] = 0;  uint offs = 0;
    for(uint i = 0; i < RADIX_MAX; i++)
    {
        uint2 res = local_scan(buf, count[i]);
        for(uint pos = index; pos < group_count; pos += UNIT_WIDTH)
            global_index[pos * RADIX_MAX + i] += offs + res.s0;
        offs += res.s1;  barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void KERNEL shuffle_data(const global uint2 *src, global uint2 *dst,
    const global uint *local_index, const global uint *global_index, uint order)
{
    const uint group = get_group_id(0);  src += group * SORT_WIDTH;
    local_index += group * SORT_WIDTH;  global_index += group * RADIX_MAX;
    const uint index = get_local_id(0);  order *= RADIX_SHIFT;
    for(uint i = 0; i < SORT_BLOCK; i++)
    {
        uint2 data = src[index * SORT_BLOCK + i];
        uint pos = global_index[(data.s0 >> order) & RADIX_MASK] + local_index[index * SORT_BLOCK + i];
        dst[pos] = data;
    }
}
