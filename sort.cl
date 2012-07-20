// sort.cl -- sorting algorithm
//


#define SORT_WIDTH  (SORT_BLOCK * UNIT_WIDTH)

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

void KERNEL local_count(const global uint2 *val, const global uint *val_count,
    global uint *local_index, global uint *global_index, uint shift, uint mask, uint max_val)
{
    const uint block = get_group_id(0), offs = block * SORT_WIDTH, n = *val_count;
    if(offs >= n)return;  val += offs;  local_index += offs;  global_index += block * RADIX_MAX;

    const uint index = get_local_id(0);  uint data[SORT_BLOCK];
    uint block_size = min((uint)SORT_BLOCK, (n - offs + (UNIT_WIDTH - 1) - index) / UNIT_WIDTH);
    for(uint i = 0; i < block_size; i++)data[i] = val[i * UNIT_WIDTH + index].s0;

    uint count[RADIX_MAX];
    for(uint i = 0; i < max_val; i++)count[i] = 0;

    uint pos[SORT_BLOCK];
    for(uint i = 0; i < block_size; i++)pos[i] = count[(data[i] >> shift) & mask]++;

    local uint buf[2 * UNIT_WIDTH];  buf[index] = 0;
    for(uint i = 0; i < max_val; i++)
    {
        uint2 res = local_scan(buf, count[i]);  count[i] = res.s0;
        if(!index)global_index[i] = res.s1;  barrier(CLK_LOCAL_MEM_FENCE);
    }
    for(uint i = 0; i < block_size; i++)
        local_index[i * UNIT_WIDTH + index] = pos[i] + count[(data[i] >> shift) & mask];
}

void KERNEL global_count(global uint *global_index, const global uint *val_count, uint max_val)  // single unit
{
    const uint index = get_global_id(0);
    uint block_count = (*val_count + SORT_WIDTH - 1) / SORT_WIDTH;
    uint2 range = block_count * (uint2)(index, index + 1) / UNIT_WIDTH;

    uint count[RADIX_MAX];
    for(uint i = 0; i < max_val; i++)
    {
        count[i] = 0;
        for(uint pos = range.s0; pos < range.s1; pos++)
        {
            uint k = pos * RADIX_MAX + i;  uint n = global_index[k];
            global_index[k] = count[i];  count[i] += n;
        }
    }

    local uint buf[2 * UNIT_WIDTH];  buf[index] = 0;  uint offs = 0;
    for(uint i = 0; i < max_val; i++)
    {
        uint2 res = local_scan(buf, count[i]);
        for(uint pos = range.s0; pos < range.s1; pos++)
            global_index[pos * RADIX_MAX + i] += offs + res.s0;
        offs += res.s1;  barrier(CLK_LOCAL_MEM_FENCE);
    }
}

uint sort_block_index(uint pos, uint n)
{
    uint local_pos = pos % SORT_WIDTH;  pos -= local_pos;  n = min((uint)SORT_WIDTH, n - pos);
    uint block_size = n / UNIT_WIDTH, rem = n % UNIT_WIDTH, bound = rem * (block_size + 1);
    if(local_pos < bound)block_size++;
    else
    {
        local_pos -= bound;  pos += rem;
    }
    return pos + local_pos / block_size + local_pos % block_size * UNIT_WIDTH;
}

void KERNEL shuffle_data(const global uint2 *src, global uint2 *dst, const global uint *val_count,
    const global uint *local_index, const global uint *global_index, uint shift, uint mask, uint last)
{
    const uint block = get_group_id(0), offs = block * SORT_WIDTH, n = *val_count;
    src += offs;  local_index += offs;  global_index += block * RADIX_MAX;

    const uint index = get_local_id(0);
    uint block_size = min((uint)SORT_BLOCK, (max(offs, n) - offs + (UNIT_WIDTH - 1) - index) / UNIT_WIDTH);
    for(uint i = 0; i < block_size; i++)
    {
        uint2 data = src[i * UNIT_WIDTH + index];
        uint pos = global_index[(data.s0 >> shift) & mask] + local_index[i * UNIT_WIDTH + index];
        dst[last ? pos : sort_block_index(pos, n)] = data;
    }
    for(uint i = block_size; i < SORT_BLOCK; i++)
    {
        uint pos = i * UNIT_WIDTH + index;  dst[offs + pos] = src[pos];
    }
}
