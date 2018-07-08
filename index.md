[Go back to andre's blog](https://andrecavalcante.github.io)

# Results

# Tools
[Pytorch](https://pytorch.org)

# Dataset
[MICCAI Grand Challenge:Prostate MR Image Segmentation 2012](https://promise12.grand-challenge.org/home/)

# Validation metrics
We use the mean slicewise "intersection over union" (IOU) over MR exams, and the IOU calculated considering the entire 3d volume occupied by the prostate gland. The slicewise IOU is a more penalizing metric. This is because IOU can only be zero or undefined when a MR slice contains no prostate tissue (a MR exam may have many of those). In this case, even a single false-positive pixel makes IOU zero (intersection = 0 and union != 0). Thus, the mean slicewise IOU can be heavely penalized. To make things worse, if the model successufully produces no false-positive, the IOU is undefined (intersection = 0 and union =0).     
  
[Go back to andre's blog](https://andrecavalcante.github.io)
