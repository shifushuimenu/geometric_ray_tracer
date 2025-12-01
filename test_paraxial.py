"""Compare with Ansys Zemax OpticStudio 2025 R1.00 Student(30)"""
import unittest
import numpy as np
import subprocess

from paraxial import *
from lens import read_lens


from plot import *
import matplotlib.pyplot as plt 

# Lens doublet with real exit pupil.
doublet_realXP = """
# Surf       	Type        	Radius        	Thickness          n_d (at 0.587562 µm)   V_d (Abbe number)        	Diameter 
   0	         0	              inf     	       200.0            1.0000000	            -1.0                        0	     	        
   1	         0	              100.0     	   2.0 	            1.5168     	            64.167                      0
   2	         0	             -100.0	           60.0	            1.0000000	            -1.0                        0
   3             1                inf              30.0             1.0000000               -1.0                        0
   4             0                40.0             2.0              1.5168                  64.167                      0
   5             0               -40.0             30.630           1.0000000               -1.0                        0
   6             0                inf              0.0              1.0000000               -1.0                        0
"""

class TestDoublet(unittest.TestCase):
   
    infile = "tmp.txt"
   
    def setUp(self):
        with open(self.infile, "w") as fh:
            fh.write(doublet_realXP)
        self.lens_sequence = read_lens(self.infile, SAG=False, lens_unit="mm")
        self.PR = ParaxialRaytracer(self.lens_sequence)

    def test1_paraxial_quantities(self):
        print("paraxial image distance=", self.PR.get_image_distance(-200.0))
        self.assertTrue(np.isclose(84.60836, self.PR.get_EFL(), rtol=1e-5))
        self.assertTrue(np.isclose(4.349419, self.PR.get_BFL(), rtol=1e-5))
        self.assertTrue(np.isclose(-0.2694382, self.PR.get_magnification(), rtol=1e-5))
        # print("angular magnification=", self.PR.get_angular_magnification())
        print("exit pupil=", self.PR.get_exit_pupil())
        print("entrance pupil=", self.PR.get_entrance_pupil())

    def tearDown(self):
        subprocess.run(["rm", self.infile])


class TestStepperLens(unittest.TestCase):
    
    def setUp(self):
        self.lens_sequence = read_lens("lens_files/stepper_lens.txt", SAG=False)
        self.PR = ParaxialRaytracer(self.lens_sequence)
    
    def test1_paraxial_quantities(self):
        pass
        # print("angular magnification=", self.PR.get_angular_magnification())
        print("exit pupil=", self.PR.get_exit_pupil())
        print("entrance pupil=", self.PR.get_entrance_pupil())
        print("EFL=", self.PR.get_EFL())
        print("BFL=", self.PR.get_BFL())

    def tearDown(self):
        pass


class TestKidger2004Lens(unittest.TestCase):
    
    def setUp(self):
        self.lens_sequence = read_lens("lens_files/lens_Kidger2004_modified.txt", SAG=False)
        self.PR = ParaxialRaytracer(self.lens_sequence)
    
    def test1_paraxial_quantities(self):
        pass
        # print("angular magnification=", self.PR.get_angular_magnification())
        print("exit pupil=", self.PR.get_exit_pupil())
        print("entrance pupil=", self.PR.get_entrance_pupil())
        print("EFL=", self.PR.get_EFL())
        print("BFL=", self.PR.get_BFL())
        # print("magnification=", self.PR.get_magnification())
        # self.assertTrue(np.isclose(84.60836, self.PR.get_EF3L(), rtol=1e-5))
        # self.assertTrue(np.isclose(4.349419, self.PR.get_BFL(), rtol=1e-5))
        # self.assertTrue(np.isclose(-0.2694382, self.PR.get_magnification(), rtol=1e-5))


    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
