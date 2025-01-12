#ifndef GAMESCOPE_EFFECTS_MANAGER_HPP
#define GAMESCOPE_EFFECTS_MANAGER_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>
#include <map>

// Vulkan device wrapper
struct GamescopeVkDevice
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;
};

// Configuration loader
class Config
{
public:
    Config() = default;

    bool loadFromFile(const std::string& path);
    std::string getString(const std::string& key, const std::string& defVal = "") const;
    float getFloat(const std::string& key, float defVal = 0.0f) const;

private:
    static void trim(std::string& s);
    std::map<std::string, std::string> m_map;
};

// Effect base class
class Effect
{
public:
    virtual ~Effect() = default;
    virtual void apply(VkCommandBuffer commandBuffer, uint32_t imageIndex) = 0;
    virtual void setupResources(VkCommandBuffer commandBuffer, uint32_t imageIndex) = 0;
};

// Example effect: LiftGammaGain
class LiftGammaGainEffect : public Effect
{
public:
    LiftGammaGainEffect(GamescopeVkDevice* device, Config& config);
    void apply(VkCommandBuffer commandBuffer, uint32_t imageIndex) override;
    void setupResources(VkCommandBuffer commandBuffer, uint32_t imageIndex) override;

private:
    void initPipeline();
    VkShaderModule createShaderModule(VkDevice device, const std::string& filepath);

    GamescopeVkDevice* m_device;
    float m_lift = 1.0f;
    float m_gamma = 1.0f;
    float m_gain = 1.0f;

    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkRenderPass m_renderPass = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;
    std::vector<VkImageView> m_imageViews;
};

// Effects manager
class GamescopeEffectsManager
{
public:
    void initialize(GamescopeVkDevice* device, const std::string& configPath);
    void applyEffects(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void destroyAll();

private:
    void addEffect(const std::string& name);

    GamescopeVkDevice* m_device;
    Config m_config;
    std::vector<std::shared_ptr<Effect>> m_effects;
};

#endif // GAMESCOPE_EFFECTS_MANAGER_HPP
