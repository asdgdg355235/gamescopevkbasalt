#include "gamescope_effects_manager.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <stdexcept>

// -----------------------------------------------------------------------------
// ASSERT_VULKAN macro for quick error checking
// -----------------------------------------------------------------------------
#ifndef ASSERT_VULKAN
#define ASSERT_VULKAN(val)                                      \
    if ((val) != VK_SUCCESS)                                    \
    {                                                           \
        throw std::runtime_error(                               \
            "Vulkan call failed at " + std::string(__FILE__) +  \
            ":" + std::to_string(__LINE__)                      \
        );                                                      \
    }
#endif

// -----------------------------------------------------------------------------
// Effects Manager Implementation
// -----------------------------------------------------------------------------
void GamescopeEffectsManager::initialize(GamescopeVkDevice* device, const std::string& configPath)
{
    m_device = device;
    if (!m_config.loadFromFile(configPath))
    {
        throw std::runtime_error("Failed to load configuration from " + configPath);
    }

    // Parse effects
    std::string effectsLine = m_config.getString("effects", "");
    auto effectNames = split(effectsLine, ':');
    for (const auto& name : effectNames)
    {
        addEffect(name);
    }
}

void GamescopeEffectsManager::applyEffects(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    for (const auto& effect : m_effects)
    {
        effect->setupResources(commandBuffer, imageIndex);
        effect->apply(commandBuffer, imageIndex);
    }
}

void GamescopeEffectsManager::addEffect(const std::string& name)
{
    if (name == "LiftGammaGain")
    {
        m_effects.push_back(std::make_shared<LiftGammaGainEffect>(m_device, m_config));
    }
    else
    {
        std::cerr << "Unknown effect: " << name << std::endl;
    }
}

std::vector<std::string> GamescopeEffectsManager::split(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::istringstream stream(str);
    std::string token;
    while (std::getline(stream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

// -----------------------------------------------------------------------------
// LiftGammaGainEffect Implementation
// -----------------------------------------------------------------------------
LiftGammaGainEffect::LiftGammaGainEffect(GamescopeVkDevice* device, Config& config)
    : m_device(device)
{
    m_lift = config.getFloat("Lift", 1.0f);
    m_gamma = config.getFloat("Gamma", 1.0f);
    m_gain = config.getFloat("Gain", 1.0f);

    initPipeline();
}

void LiftGammaGainEffect::apply(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    // Bind pipeline and draw
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
}

void LiftGammaGainEffect::setupResources(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    // Update descriptor sets dynamically based on image index
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = m_sampler;
    imageInfo.imageView = m_imageViews[imageIndex];
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = m_descriptorSet;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrite.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(m_device->device, 1, &descriptorWrite, 0, nullptr);
}

void LiftGammaGainEffect::initPipeline()
{
    // Create a render pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_B8G8R8A8_UNORM;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    ASSERT_VULKAN(vkCreateRenderPass(m_device->device, &renderPassInfo, nullptr, &m_renderPass));

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ASSERT_VULKAN(vkCreatePipelineLayout(m_device->device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));

    // Load shaders and create pipeline
    VkShaderModule vertexShader = createShaderModule(m_device->device, "shaders/lift_gamma_gain.vert.spv");
    VkShaderModule fragmentShader = createShaderModule(m_device->device, "shaders/lift_gamma_gain.frag.spv");

    VkPipelineShaderStageCreateInfo shaderStages[2] = {};

    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertexShader;
    shaderStages[0].pName = "main";

    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragmentShader;
    shaderStages[1].pName = "main";

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.renderPass = m_renderPass;
    pipelineInfo.layout = m_pipelineLayout;

    ASSERT_VULKAN(vkCreateGraphicsPipelines(m_device->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline));

    vkDestroyShaderModule(m_device->device, vertexShader, nullptr);
    vkDestroyShaderModule(m_device->device, fragmentShader, nullptr);
}

VkShaderModule LiftGammaGainEffect::createShaderModule(VkDevice device, const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filepath);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    VkShaderModule shaderModule;
    ASSERT_VULKAN(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));

    return shaderModule;
}
