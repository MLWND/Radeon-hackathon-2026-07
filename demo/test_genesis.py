"""
RoboPilot Genesis Test
验证 Genesis 仿真环境正常工作
"""
import genesis as gs

gs.init()
scene = gs.Scene()
scene.build()
print("Genesis OK - Scene built successfully")
