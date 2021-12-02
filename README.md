# ImageProcessing-hafman_encoding
Load the image from the file and convert it into grey-scale, 8bpp format

Compute and display the image histogram

Change the image's contrast such that 20% of the pixels have grey levels between 0-75,
50% of the pixels to have grey levels between 75-200 and balance pixels to have greylevel over 201. Display the new image and its histogram.

Subject the image to median-filtering with a 3x3 window

Sharpen the image with a sharpening filter having a 3x3 window.

Compute the difference between the current image and its original form (i.e. loaded from the file) in the RMS error.

Compute and display the image entropy and determine the best possible coding efficiency in terms of average bits per pixel.

Encode the image using a variable-length code based on the Huffman coding algorithm

Save the encoded file. Write a separate function that can load the encoded file and display it on the screen.