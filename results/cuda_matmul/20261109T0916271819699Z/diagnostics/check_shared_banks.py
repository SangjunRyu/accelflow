from collections import defaultdict
import json

# Source-level scalar FP32 model. Same-word reads are broadcasts, not conflicts.
patterns = {
    'current_store_A_or_B': lambda x,y,k: 16*y+x,
    'current_read_A': lambda x,y,k: 16*y+k,
    'current_read_B': lambda x,y,k: 16*k+x,
    'blind_padding17_store': lambda x,y,k: 17*y+x,
    'xor16_store': lambda x,y,k: 16*y+(x^(y&15)),
    'xor16_read_A': lambda x,y,k: 16*y+(k^(y&15)),
    'xor16_read_B': lambda x,y,k: 16*k+(x^(k&15)),
}
result={}
for name,address in patterns.items():
    maximum=0
    for warp in range(8):
        for k in range(16):
            banks=defaultdict(set)
            for lane in range(32):
                linear=warp*32+lane
                word=address(linear%16,linear//16,k)
                banks[word%32].add(word)
            maximum=max(maximum,max(map(len,banks.values())))
    result[name]=maximum
assert all(result[x]==1 for x in ('current_store_A_or_B','current_read_A','current_read_B'))
assert result['blind_padding17_store']==2
print(json.dumps({'model':'32 banks, scalar FP32, bank=word_index%32; distinct words per bank',
                  'all_8_warps_and_16_inner_positions_checked':True,
                  'maximum_distinct_words_in_one_bank_per_request':result},indent=2))
