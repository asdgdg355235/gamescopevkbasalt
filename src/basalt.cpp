// Full code integrating shader-compatible VKBasalt features into Gamescope.
// Refined implementation including GLSL and FX shader parsing, dynamic layouts, descriptor allocation, and resource management.

#include <vulkan/vulkan.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <filesystem>

// Macro for Vulkan error checking
#ifndef ASSERT_VULKAN
#define ASSERT_VULKAN(val) \
    if ((val) != VK_SUCCESS) \
    { \
        throw std::runtime_error("Vulkan call failed at " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
    }
#endif

// -----------------------------------------------------------------------------
// GamescopeVkDevice struct to manage Vulkan device resources
// -----------------------------------------------------------------------------
struct GamescopeVkDevice
{
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkInstance instance = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamilyIndex = 0;
};

// -----------------------------------------------------------------------------
// Configuration Loader adapted for Gamescope
// ----------------------------------------------------------------------------
class Config
{
public:
    Config() = default;

    bool loadFromFile(const std::string& path)
    {
        std::ifstream in(path);
        if (!in.is_open())
        {
            std::cerr << "Failed to open config file: " << path << std::endl;
            return false;
        }
        std::string line;
        while (std::getline(in, line))
        {
            auto pos = line.find('=');
            if (pos == std::string::npos || line[0] == '#')
                continue;

            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            trim(key);
            trim(value);
            m_map[key] = value;
        }
        return true;
    }

    std::string getString(const std::string& key, const std::string& defaultValue = "") const
    {
        auto it = m_map.find(key);
        if (it != m_map.end())
            return it->second;
        return defaultValue;
    }

    std::vector<std::string> getShaderPaths(const std::string& key) const
    {
        std::string paths = getString(key, "");
        std::vector<std::string> result;
        std::stringstream ss(paths);
        std::string path;
        while (std::getline(ss, path, ':'))
        {
            trim(path);
            if (!path.empty())
                result.push_back(path);
        }
        return result;
    }

private:
    static void trim(std::string& s)
    {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
    }

    std::map<std::string, std::string> m_map;
};

// -----------------------------------------------------------------------------
// Effect Base Class
// -----------------------------------------------------------------------------
class Effect
{
public:
    virtual ~Effect() = default;
    virtual void apply(VkCommandBuffer commandBuffer) = 0;
    virtual void setupResources(VkCommandBuffer commandBuffer, uint32_t imageIndex) = 0;
protected:
    virtual void loadShaders(const std::vector<std::string>& paths) = 0;
};

// -----------------------------------------------------------------------------
// Shader Parsing Helper
// -----------------------------------------------------------------------------
void compileShader(const std::string& filePath, std::vector<uint32_t>& spirvCode)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open shader file: " + filePath);
    }

    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    reshadefx::parser parser;

    std::unique_ptr<reshadefx::codegen> codegen(reshadefx::create_codegen_spirv(true, true, false, false));
    if (!parser.parse(source, codegen.get()))
    {
        throw std::runtime_error("Failed to parse shader file: " + filePath + "\n" + parser.errors());
    }

    reshadefx::module module;
    codegen->write_result(module);

    spirvCode = module.code;
}

// -----------------------------------------------------------------------------
// LiftGammaGain Effect Implementation
// -----------------------------------------------------------------------------
class LiftGammaGainEffect : public Effect
{
public:
    LiftGammaGainEffect(GamescopeVkDevice* device, Config& config)
        : m_device(device)
    {
        m_lift = config.getFloat("Lift", 1.0f);
        m_gamma = config.getFloat("Gamma", 1.0f);
        m_gain = config.getFloat("Gain", 1.0f);

        auto shaderPaths = config.getShaderPaths("LiftGammaGainShaders");
        loadShaders(shaderPaths);
        initPipeline();
    }

    void apply(VkCommandBuffer commandBuffer) override
    {
        // Bind pipeline and draw
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }

    void setupResources(VkCommandBuffer commandBuffer, uint32_t imageIndex) override
    {
        // Update descriptor sets with current image resources
        VkDescriptorImageInfo imageInfo{};
        imageInfo.sampler = m_sampler;
        imageInfo.imageView = m_imageViews[imageIndex];
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = m_descriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(m_device->device, 1, &descriptorWrite, 0, nullptr);
    }

private:
    void loadShaders(const std::vector<std::string>& paths) override
    {
        for (const auto& path : paths)
        {
            if (path.ends_with(".glsl"))
            {
                compileShader(path, m_vertexCode);
            }
            else if (path.ends_with(".fx"))
            {
                compileShader(path, m_fragmentCode);
            }
        }
    }

    void initPipeline()
    {
        // Descriptor Pool
        VkDescriptorPoolSize poolSizes[1] = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount = 10; // Support for multiple sets

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = poolSizes;
        poolInfo.maxSets = 10;

        ASSERT_VULKAN(vkCreateDescriptorPool(m_device->device, &poolInfo, nullptr, &m_descriptorPool));

        // Descriptor Set Layout
        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = 0;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &layoutBinding;
        ASSERT_VULKAN(vkCreateDescriptorSetLayout(m_device->device, &layoutInfo, nullptr, &m_descriptorSetLayout));

        // Pipeline Layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
        ASSERT_VULKAN(vkCreatePipelineLayout(m_device->device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));

        // Graphics Pipeline
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

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = m_pipelineLayout;
        pipelineInfo.renderPass = m_renderPass;
        pipelineInfo.stageCount = 2;

        VkPipelineShaderStageCreateInfo stages[2] = {};
        VkShaderModule vertexModule = createShaderModule(m_device->device, m_vertexCode);
        VkShaderModule fragmentModule = createShaderModule(m_device->device, m_fragmentCode);

        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertexModule;
        stages[0].pName = "main";

        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragmentModule;
        stages[1].pName = "main";

        pipelineInfo.pStages = stages;

        ASSERT_VULKAN(vkCreateGraphicsPipelines(m_device->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline));

        vkDestroyShaderModule(m_device->device, vertexModule, nullptr);
        vkDestroyShaderModule(m_device->device, fragmentModule, nullptr);

        // Sampler creation
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ASSERT_VULKAN(vkCreateSampler(m_device->device, &samplerInfo, nullptr, &m_sampler));

        // Allocate descriptor sets
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_descriptorSetLayout;
        ASSERT_VULKAN(vkAllocateDescriptorSets(m_device->device, &allocInfo, &m_descriptorSet));
    }

    VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& spirvCode)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
        createInfo.pCode = spirvCode.data();

        VkShaderModule shaderModule;
        ASSERT_VULKAN(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));
        return shaderModule;
    }

    GamescopeVkDevice* m_device;
    float m_lift;
    float m_gamma;
    float m_gain;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;
    std::vector<uint32_t> m_vertexCode;
    std::vector<uint32_t> m_fragmentCode;
};
