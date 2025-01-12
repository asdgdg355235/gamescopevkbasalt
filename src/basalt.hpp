#ifndef BASALT_HPP
#define BASALT_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>

// -----------------------------------------------------------------------------
// Vulkan device management structure
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
// Configuration loader for parsing config files
// -----------------------------------------------------------------------------
class Config
{
public:
    Config() = default;

    bool loadFromFile(const std::string& path);
    std::string getString(const std::string& key, const std::string& defaultValue = "") const;
    std::vector<std::string> getShaderPaths(const std::string& key) const;

private:
    static void trim(std::string& s);
    std::map<std::string, std::string> m_map;
};

// -----------------------------------------------------------------------------
// Base class for effects
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
// LiftGammaGainEffect class inheriting from Effect
// -----------------------------------------------------------------------------
class LiftGammaGainEffect : public Effect
{
public:
    LiftGammaGainEffect(GamescopeVkDevice* device, Config& config);

    void apply(VkCommandBuffer commandBuffer) override;
    void setupResources(VkCommandBuffer commandBuffer, uint32_t imageIndex) override;

private:
    void loadShaders(const std::vector<std::string>& paths) override;
    void initPipeline();
    VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& spirvCode);

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

// -----------------------------------------------------------------------------
// Shader utility functions
// -----------------------------------------------------------------------------
void compileShader(const std::string& filePath, std::vector<uint32_t>& spirvCode);

#endif // BASALT_HPP
