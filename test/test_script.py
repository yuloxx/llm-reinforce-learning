from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from typing import List, Optional
import unittest


class ScriptRenderTest(unittest.TestCase):
    # test virtual home gather food environment
    YOUR_FILE_NAME = "D:\\programs\\windows_exec.v2.2.4\\VirtualHome.exe"

    def setUp(self):
        self.comm = UnityCommunication(file_name=self.YOUR_FILE_NAME, timeout_wait=180)

    def tearDown(self):
        self.comm.close()

    def render_skeleton(self, env: int, script: List[str], batch_size: Optional[int] = None) -> bool:
        self.comm.reset(env)
        self.comm.add_character()

        if batch_size is None or batch_size <= 0:
            res, _ = self.comm.render_script(script, recording=True, skip_animation=False)
            return res

        for i in range(0, len(script), batch_size):
            batch = script[i:i + batch_size]
            res, _ = self.comm.render_script(batch, recording=True, skip_animation=False)
            if not res:
                return False
        return True

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

    def test_script_env1_grab_food_one_by_one(self):
        script = ['<char0> [walk] <chicken> (165)', '<char0> [grab] <chicken> (165)', '<char0> [walk] <fridge> (149)',
                  '<char0> [open] <fridge> (149)', '<char0> [putin] <chicken> (165) <fridge> (149)',
                  '<char0> [close] <fridge> (149)', '<char0> [walk] <chocolatesyrup> (183)',
                  '<char0> [grab] <chocolatesyrup> (183)', '<char0> [walk] <fridge> (149)',
                  '<char0> [open] <fridge> (149)', '<char0> [putin] <chocolatesyrup> (183) <fridge> (149)',
                  '<char0> [close] <fridge> (149)']

        res = self.render_skeleton(env=1, script=script)
        self.assertTrue(res)

    def test_script_env2_grab_food_one_by_one(self):
        script = ['<char0> [walk] <salmon> (182)', '<char0> [grab] <salmon> (182)', '<char0> [walk] <fridge> (163)',
                  '<char0> [open] <fridge> (163)', '<char0> [putin] <salmon> (182) <fridge> (163)',
                  '<char0> [close] <fridge> (163)', '<char0> [walk] <mincedmeat> (183)',
                  '<char0> [grab] <mincedmeat> (183)', '<char0> [walk] <fridge> (163)', '<char0> [open] <fridge> (163)',
                  '<char0> [putin] <mincedmeat> (183) <fridge> (163)', '<char0> [close] <fridge> (163)',
                  '<char0> [walk] <juice> (175)', '<char0> [grab] <juice> (175)', '<char0> [walk] <fridge> (163)',
                  '<char0> [open] <fridge> (163)', '<char0> [putin] <juice> (175) <fridge> (163)',
                  '<char0> [close] <fridge> (163)']
        res = self.render_skeleton(env=2, script=script)
        self.assertTrue(res)

    def test_script_env3_grab_food_one_by_one(self):
        script = ['<char0> [walk] <pancake> (62)', '<char0> [grab] <pancake> (62)', '<char0> [walk] <fridge> (103)',
                  '<char0> [open] <fridge> (103)', '<char0> [putin] <pancake> (62) <fridge> (103)',
                  '<char0> [close] <fridge> (103)', '<char0> [walk] <pear> (64)', '<char0> [grab] <pear> (64)',
                  '<char0> [walk] <fridge> (103)', '<char0> [open] <fridge> (103)',
                  '<char0> [putin] <pear> (64) <fridge> (103)', '<char0> [close] <fridge> (103)',
                  '<char0> [walk] <milkshake> (324)', '<char0> [grab] <milkshake> (324)',
                  '<char0> [walk] <fridge> (103)', '<char0> [open] <fridge> (103)',
                  '<char0> [putin] <milkshake> (324) <fridge> (103)', '<char0> [close] <fridge> (103)',
                  '<char0> [walk] <chocolatesyrup> (57)', '<char0> [grab] <chocolatesyrup> (57)',
                  '<char0> [walk] <fridge> (103)', '<char0> [open] <fridge> (103)',
                  '<char0> [putin] <chocolatesyrup> (57) <fridge> (103)', '<char0> [close] <fridge> (103)']
        res = self.render_skeleton(env=3, script=script, batch_size=10)
        self.assertTrue(res)

    def test_script_env4_grab_food_one_by_one(self):
        script = ['<char0> [walk] <pie> (211)', '<char0> [grab] <pie> (211)', '<char0> [walk] <fridge> (155)',
                  '<char0> [open] <fridge> (155)', '<char0> [putin] <pie> (211) <fridge> (155)',
                  '<char0> [close] <fridge> (155)', '<char0> [walk] <mincedmeat> (204)',
                  '<char0> [grab] <mincedmeat> (204)', '<char0> [walk] <fridge> (155)', '<char0> [open] <fridge> (155)',
                  '<char0> [putin] <mincedmeat> (204) <fridge> (155)', '<char0> [close] <fridge> (155)',
                  '<char0> [walk] <chocolatesyrup> (206)', '<char0> [grab] <chocolatesyrup> (206)',
                  '<char0> [walk] <fridge> (155)', '<char0> [open] <fridge> (155)',
                  '<char0> [putin] <chocolatesyrup> (206) <fridge> (155)', '<char0> [close] <fridge> (155)']
        res = self.render_skeleton(env=4, script=script, batch_size=10)
        self.assertTrue(res)

    def test_script_env5_grab_food_one_by_one(self):
        script = ['<char0> [walk] <salad> (256)', '<char0> [grab] <salad> (256)', '<char0> [walk] <fridge> (247)',
                  '<char0> [open] <fridge> (247)', '<char0> [putin] <salad> (256) <fridge> (247)',
                  '<char0> [close] <fridge> (247)', '<char0> [walk] <carrot> (255)', '<char0> [grab] <carrot> (255)',
                  '<char0> [walk] <fridge> (247)', '<char0> [open] <fridge> (247)',
                  '<char0> [putin] <carrot> (255) <fridge> (247)', '<char0> [close] <fridge> (247)']
        res = self.render_skeleton(env=5, script=script, batch_size=10)
        self.assertTrue(res)

    def test_script_env6_grab_food_one_by_one(self):
        script = ['<char0> [walk] <wine> (335)', '<char0> [grab] <wine> (335)', '<char0> [walk] <fridge> (166)',
                  '<char0> [open] <fridge> (166)', '<char0> [putin] <wine> (335) <fridge> (166)',
                  '<char0> [close] <fridge> (166)', '<char0> [walk] <juice> (333)', '<char0> [grab] <juice> (333)',
                  '<char0> [walk] <fridge> (166)', '<char0> [open] <fridge> (166)',
                  '<char0> [putin] <juice> (333) <fridge> (166)', '<char0> [close] <fridge> (166)',
                  '<char0> [walk] <chocolatesyrup> (193)', '<char0> [grab] <chocolatesyrup> (193)',
                  '<char0> [walk] <fridge> (166)', '<char0> [open] <fridge> (166)',
                  '<char0> [putin] <chocolatesyrup> (193) <fridge> (166)', '<char0> [close] <fridge> (166)',
                  '<char0> [walk] <chicken> (173)', '<char0> [grab] <chicken> (173)', '<char0> [walk] <fridge> (166)',
                  '<char0> [open] <fridge> (166)', '<char0> [putin] <chicken> (173) <fridge> (166)',
                  '<char0> [close] <fridge> (166)']

        res = self.render_skeleton(env=6, script=script, batch_size=10)
        self.assertTrue(res)




if __name__ == '__main__':
    unittest.main()
