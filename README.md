# [ICCV 2025] Textured 3D Regenerative Morphing with 3D Diffusion Prior

## Instructions for Running Morphing

1. **Set up the environment**  
   Follow the instructions in **[GaussianAnything](https://github.com/NIRVANALAN/GaussianAnything)** to configure the environment and install all dependencies.

2. **Prepare 3D objects and train LoRA**  
   Select two 3D objects as source and target. Use the training code in **[GaussianAnything](https://github.com/NIRVANALAN/GaussianAnything)** to train a LoRA model.

3. **Configure LoRA paths**  
   Open **`morphing.sh`** and **`morphing.py`**, and update the file paths to point to your trained LoRA checkpoints.

4. **Run morphing**  
   Execute the **`morphing.sh`** script to launch the morphing process and inspect the generated results.

