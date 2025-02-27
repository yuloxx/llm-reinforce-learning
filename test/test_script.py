from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from typing import List
import unittest


class ScriptRenderTest(unittest.TestCase):
    # test virtual home gather food environment
    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"

    def setUp(self):
        self.comm = UnityCommunication(file_name=self.YOUR_FILE_NAME)

    def tearDown(self):
        self.comm.close()

    def render_skeleton(self, env: int, script: List[str]) -> bool:
        self.comm.reset(env)
        self.comm.add_character()
        res, _ = self.comm.render_script(script, recording=True, skip_animation=False)
        return res

    def test_script_env0_grab_food_one_by_one(self):
        script = ['<char0> [walk] <salmon> (328)', '<char0> [grab] <salmon> (328)', '<char0> [walk] <fridge> (306)',
                  '<char0> [open] <fridge> (306)', '<char0> [putin] <salmon> (328) <fridge> (306)',
                  '<char0> [close] <fridge> (306)', '<char0> [walk] <pie> (320)', '<char0> [grab] <pie> (320)',
                  '<char0> [walk] <fridge> (306)', '<char0> [open] <fridge> (306)',
                  '<char0> [putin] <pie> (320) <fridge> (306)', '<char0> [close] <fridge> (306)',
                  '<char0> [walk] <chocolatesyrup> (332)', '<char0> [grab] <chocolatesyrup> (332)',
                  '<char0> [walk] <fridge> (306)', '<char0> [open] <fridge> (306)',
                  '<char0> [putin] <chocolatesyrup> (332) <fridge> (306)', '<char0> [close] <fridge> (306)']
        res = self.render_skeleton(env=0, script=script)
        self.assertTrue(res)

    def test_script_env0_grab_food_fullhands(self):
        script = ['<char0> [walk] <salmon> (328)', '<char0> [grab] <salmon> (328)', '<char0> [walk] <fridge> (306)',
                  '<char0> [open] <fridge> (306)', '<char0> [putin] <salmon> (328) <fridge> (306)',
                  '<char0> [walk] <pie> (320)', '<char0> [grab] <pie> (320)', '<char0> [walk] <chocolatesyrup> (332)',
                  '<char0> [grab] <chocolatesyrup> (332)', '<char0> [walk] <fridge> (306)',
                  '<char0> [putin] <pie> (320) <fridge> (306)', '<char0> [putin] <chocolatesyrup> (332) <fridge> (306)',
                  '<char0> [close] <fridge> (306)']
        res = self.render_skeleton(env=0, script=script)
        self.assertTrue(res)

    def test_script_env0_put_food_on_fridge(self):
        script = ['<char0> [walk] <salmon> (328)', '<char0> [grab] <salmon> (328)', '<char0> [walk] <fridge> (306)',
                  '<char0> [put] <salmon> (328) <fridge> (306)']
        res = self.render_skeleton(env=0, script=script)
        self.assertTrue(res)

    def test_script_env0_put_food_on_coffeetable(self):
        script = ['<char0> [walk] <salmon> (328)', '<char0> [grab] <salmon> (328)',
                  '<char0> [walk] <coffeetable> (372)', '<char0> [put] <salmon> (328) <coffeetable> (372)']
        res = self.render_skeleton(env=0, script=script)
        self.assertTrue(res)

    def test_script_env0_grab_food_inside_fridge(self):
        script = ['<char0> [walk] <salmon> (328)', '<char0> [grab] <salmon> (328)', '<char0> [walk] <fridge> (306)',
                  '<char0> [open] <fridge> (306)', '<char0> [putin] <salmon> (328) <fridge> (306)',
                  '<char0> [grab] <salmon> (328)']
        res = self.render_skeleton(env=0, script=script)
        self.assertTrue(res)

    def test_script_env1_grab_chicken(self):
        script = ['<char0> [walk] <chicken> (165)', '<char0> [grab] <chicken> (165)', '<char0> [walk] <fridge> (149)',
                  '<char0> [open] <fridge> (149)', '<char0> [putin] <chicken> (165) <fridge> (149)',
                  '<char0> [close] <fridge> (149)']
        res = self.render_skeleton(env=1, script=script)
        self.assertTrue(res)

    def test_script_env1_x(self):
        script = ['<char0> [walk] <chicken> (165)', '<char0> [grab] <chicken> (165)',
                  '<char0> [walk] <chocolatesyrup> (183)', '<char0> [grab] <chocolatesyrup> (183)',
                  '<char0> [walk] <fridge> (149)']
        res = self.render_skeleton(env=1, script=script)
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()
