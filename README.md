# Pix2pix implementation (Map to aerial photo) 
09/02/2022

While working on paired style transfer: \
Implementation of the [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) paper 
by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros with the help of educational resources. 

The model translates map pictures to the corresponding aerial pictures. \
Made with Python and Keras.

### Results on test data: 

<img width="763" alt="Screen Shot 2022-09-02 at 6 17 16 PM" src="https://user-images.githubusercontent.com/37712544/188196250-8a859c97-68cf-436c-bb78-8eae68502018.png">
<img width="763" alt="Screen Shot 2022-09-02 at 6 16 44 PM" src="https://user-images.githubusercontent.com/37712544/188196261-51dd8f71-1e5e-4523-8d04-e5d00897fa11.png">

<br>

The model uses:
- Conditional GANs
- a PatchGAN discriminator, which only penalizes structure at the scale of image patches rather than the whole image.
- a U-Net generator (encoder-decoder architecture with skip connections)




