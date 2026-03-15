import sapien.core as sapien

# 初始化 SAPIEN 引擎
engine = sapien.Engine()
renderer = sapien.VulkanRenderer()
engine.set_renderer(renderer)

# 打印当前渲染器正在使用的物理设备名称
print("当前 SAPIEN 使用的渲染设备是:", renderer._internal_context.get_device_name())
