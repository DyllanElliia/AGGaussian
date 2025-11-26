/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>

#include <cstdio>
#include <string>
#include <tuple>

// [rendered, out_color, out_semantic, out_depth, out_alpha, *out_distortion*,
// radii, geomBuffer, binningBuffer, imgBuffer]
std::tuple<
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &colors,
    const torch::Tensor &semantics,
    const torch::Tensor &opacity,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const bool prefiltered,
    const bool debug,
    const bool render_ellipsoid);

// [rendered, out_color, gau_sem, num_gsem, geomBuffer, binningBuffer,
// imgBuffer]
std::tuple<
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
TraceGaussiansCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &colors,
    const torch::Tensor &img_sem,
    const torch::Tensor &opacity,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const bool prefiltered,
    const bool debug,
    const bool render_ellipsoid);

// [dL_dmeans2D, dL_dcolors, dL_dsemantics, dL_dopacity, dL_dmeans3D, dL_dcov3D,
// dL_dsh, dL_dscales, dL_drotations]
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &background,
    const torch::Tensor &means3D,
    const torch::Tensor &radii,
    const torch::Tensor &colors,
    const torch::Tensor &semantics,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const float scale_modifier,
    const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor &dL_dout_color,
    const torch::Tensor &dL_dout_semantic,
    const torch::Tensor &dL_dout_depth,
    const torch::Tensor &dL_dout_alpha,
    const torch::Tensor &dL_dout_distortion,  // TODO: add depth and feature dist
    const torch::Tensor &sh,
    const int degree,
    const torch::Tensor &campos,
    const torch::Tensor &geomBuffer,
    const int R,
    const torch::Tensor &binningBuffer,
    const torch::Tensor &imageBuffer,
    const torch::Tensor &alphas,
    const bool debug);

torch::Tensor
markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix, torch::Tensor &projmatrix);