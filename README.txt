<<<"UEC FOOD 256": 256-kind food dataset (release 1.1)>>>

The dataset "UEC FOOD 256" contains 256-kind food photos, each of which 
has a bounding box indicating the location of the food item in the photo.

Most of the food categories in this dataset are popular foods in Japan and
other countries.  Therefore, some catarogies might not be familiar with other
people than Japanese. 

This dataset was built to implement a practical food recognition system .

In fact, we have released a real-time 256-kind food recognition application
for Android smartphones. You can get to know the detail and download an APK
file from http://foodcam.mobi/ .

This ZIP file contains the following files:

  [1-256]        :  directory names correspond to food ID.
  [1-256]/*.jpg  :  food photo files (some photos are duplicated in two or
                    more directories, since they includes two or more food items.)
  [1-256]/bb_info.txt:  bounding box information for the photo files in each directory

  category.txt    : food list including the correspondences between food
                    IDs and food names in English

If you have comments and questions, please feel free to send e-mail 
to the following address:
  food-group@mm.cs.uec.ac.jp

------------ 
Contact:

Food Recognition Research Group:
  Kawano Yoshiyuki (Master Student)
  Prof. Keiji Yanai

  food-group@mm.cs.uec.ac.jp

Department of Informatics, 
The University of Electro-Communications, Tokyo, Japan

----------------------
2015.4.12 Revised ZIP file (Redundant images are removed.)
