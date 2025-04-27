# Urban Blossoms 🌸

## Problem Statement
Graffiti diminishes the cleanliness and beauty of public spaces. When these shared areas are defaced, they become less welcoming and enjoyable. Citizens deserve to take pride in their city and the spaces they share. Beyond the aesthetic cost, graffiti can also pose real dangers. Gang-tagging, or graffiti related to gang activity, can contribute to an environment of fear and insecurity.

While one solution is to remove graffiti, it often becomes a temporary fix; many spaces are quickly defaced again. Instead, by bringing local culture and pride into these public areas with meaningful art, we want to create a sense of ownership that discourages vandalism and helps protect these spaces over time.

## Our Solution
Instead of simply erasing graffiti, we set out to transform it. Using artificial intelligence, we detect graffiti in an image and generate mural designs inspired by community input.

Our system identifies where graffiti appears and replaces it with new artwork based on prompts submitted by users. A selection of these designs is then voted on by the community, with the winning ideas commissioned for local artists and residents to bring to life.

Our goal isn’t to replace artists — it’s to empower them, giving new meaning and pride to public spaces.

## Tech Stack
**Frontend:**
- Gradio for a simple, intuitive user interface.

**Backend:**
- Graffiti Detection: Fine-tuned YOLO v8 model for bounding box detection.  
- Mask Creation: OpenCV to create black-and-white masks.  
- Inpainting: Stable Diffusion v3 Inpainting API.  
- Annotations: Labelme for data annotations.

## How It Works
**Detection:**  
We use a YOLO model trained specifically on graffiti datasets. It outputs bounding boxes around detected graffiti. We used a dataset of roughly 20,000 images of graffiti to train the model.

**Masking:**  
A black mask (0) is created with white rectangles (255) over the detected graffiti regions in the user-inputted image.

**Inpainting:**  
Both the original image and mask, along with the user's prompt, are sent to the Stable Diffusion API. It paints new mural art only where the graffiti was detected.

**Output:**  
The final image is seamlessly blended, preserving the wall's texture while showcasing the new mural.

## Challenges
- **Bounding Box Accuracy:** Graffiti comes in unpredictable shapes. Bounding boxes sometimes capture extra background.  
- **Prompt Sensitivity:** The quality of inpainting depends on how detailed and realistic the prompt is.  
- **Mask Precision:** White rectangles aren't perfect masks.  
- **Deployment:** It was quite difficult integrating the Gradio frontend with the backend APIs.

## Accomplishments
- Built an end-to-end AI pipeline that detects graffiti and generates mural designs based on community input.  
- Integrated Stable Diffusion's inpainting technology to create realistic, site-specific mural art that blends seamlessly with real-world surfaces.  
- Designed a community engagement model where users can vote on mural concepts, connecting AI creativity with local voices.  
- Created a proof-of-concept Gradio app allowing users to upload images, submit prompts, and visualize mural transformations easily.

## Next Steps
- Allow users to post their chosen image to a community platform of choice, inviting local artists to bring the image to life.  
- Implement Augmented Reality functionality to view murals in real time.  
- Integrate into the City of San Jose’s citizen reporting app.
