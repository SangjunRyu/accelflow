#pragma once

// Local workaround for the pinned Wormhole LLK's TEN-3868 trailing SrcA unpack.
// Only that discarded unpack is temporarily TF32. Actual CB payload goes straight
// to FP32 Dest. Upstream files and simulator validation are not modified.
#if defined(TRISC_UNPACK) && defined(ACCELFLOW_FP32_DUMMY_WORKAROUND)
#define unpack_to_dest_tile_done accelflow_original_unpack_to_dest_tile_done
#include "api/compute/common.h"
#undef unpack_to_dest_tile_done

namespace ckernel::unpacker {
inline void unpack_to_dest_tile_done(uint32_t& context, const uint32_t format) {
    const bool fp32 = format == static_cast<uint32_t>(DataFormat::Float32);
    if (fp32) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(static_cast<uint32_t>(DataFormat::Tf32));
    }
    accelflow_original_unpack_to_dest_tile_done(context, format);
    if (fp32) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(format);
    }
}
} // namespace ckernel::unpacker
#else
#include "api/compute/common.h"
#endif
