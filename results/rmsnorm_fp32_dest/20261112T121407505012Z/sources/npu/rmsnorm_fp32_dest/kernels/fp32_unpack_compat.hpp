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
    // Adapted from TT-Metal's cunpack_common.h (Apache-2.0, Tenstorrent AI ULC).
    // Preserve TEN-3868's trailing SrcA unpack, but use a legal TF32 discarded write
    // instead of ZeroWrite2, which TTSim refuses as UntestedFunctionality.
    t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST);
    TT_SETDMAREG(0, LOWER_HALFWORD(canonical_unpA_z_stride(format) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT),
                 0, LO_16(p_gpr_unpack::TMP_LO));
    TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    if (context == 0) {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(4 * 16);
    } else {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(4 * 16);
    }
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);
    TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXX(p_setadc::UNP_A, FACE_C_DIM - 1, 0);
    TT_UNPACR(SrcA, 0, 0, 0 /* discarded dummy uses supported config context 0 */, 0, 1, 0, p_unpacr::RAREFYB_DISABLE,
              0 /* discarded real write, not ZeroWrite2 */, 0, 0, 0, 1);
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0);
    if (fp32) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(format);
    }
}
} // namespace ckernel::unpacker
#else
#include "api/compute/common.h"
#endif
