## Seam Carving for Content-Aware Image Resizing
Will Emmanuel

Implementation of Shai Avidan and Ariel Shamir's [research](http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html)

### Requirements
* Run in provided Vagrant box
* OpenCV and NumPy required

### Instructions
* Simple Image Seam Cropping
  python ./seam_carving "filename" x y
  e.g python ./seam_carving "images/fig5.png" 5 0 to resize fig5 by 5 columns
