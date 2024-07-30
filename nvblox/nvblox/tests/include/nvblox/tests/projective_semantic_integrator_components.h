#pragma once
#include "nvblox/integrators/projective_semantic_integrator.h"
#include "nvblox/core/color.h"
#include <string>

namespace nvblox {
namespace test_utils {
ProjectiveSemanticIntegrator::SemanticConfig getSemanticConfig();

void readImageLabelOnGPU(
    const ProjectiveSemanticIntegrator::SemanticConfig& semantic_config,
    const Color& color);

void readColorFromLabel(
    const ProjectiveSemanticIntegrator::SemanticConfig& semantic_config,
    const SemanticLabel& label);

void readLabelFromImage(
    const ProjectiveSemanticIntegrator::SemanticConfig& semantic_config,
    const ColorImage& frame);


}
}