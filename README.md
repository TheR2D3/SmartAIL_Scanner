# SmartAIL_Scanner

1. This is a flask app.
2. Paste the input of image in b64 encoded format once the app is started.
3. Wait for 3-5 mins for the processing to complete.

#Processing
1. Edges of images are detected using Sobel filters.
2. Edge image is converted to a binary format.
3. The edge image is passed to Hough transform and mapped in Hough space.
4. The edge image and Hough space are displayed in output html file.

#Work in progress
1. Find the points of interest from Hough space and map it to cartesian space.
2. Find all possible quadrilaterals formed in cartesian space.
3. Find the vertices of best fitting quadrilateral in cartesian space.
4. Extract the portion quadrilateral from the original image, and transform it as the final result.
