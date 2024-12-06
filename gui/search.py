from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import open_clip


def perform_search(term, image_path, use_pc):
    # Dummy search function, replace with actual search logic
    df = pd.read_pickle('image_embeddings.pickle')
    model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    

    if term and not image_path:
        print("***TERM PROVIDED***")
        
        model.eval()
        
        text = tokenizer([term])
        query_embedding = F.normalize(model.encode_text(text))
        
        # Calculate cosine similarity between query_embedding and all embeddings in df
        cosine_similarities = df['embedding'].apply(lambda x: np.dot(query_embedding.detach().numpy(), np.array(x)) / (np.linalg.norm(query_embedding.detach().numpy()) * np.linalg.norm(np.array(x))))

        # Get the indices of the top 5 maximum cosine similarities
        top_indices = cosine_similarities.sort_values(ascending=False).head(5).index
        
        # Retrieve the image paths with the highest cosine similarities
        return_image_paths = df.loc[top_indices, 'file_name'].apply(lambda x: os.path.join('./coco_images_resized', x)).tolist()

        # print("*****" + str(return_image_paths))
        return return_image_paths
        
    if image_path and not term:
        print("***IMAGE PATH PROVIDED***")
        # Add logic to handle image search

        if use_pc:
            
            # TODO: Implement image search using principal components
            return 1
            
            
            
            
        else:
                                
            image = preprocess(Image.open(image_path)).unsqueeze(0)
            
            query_embedding = F.normalize(model.encode_image(image))
                        
            cosine_similarities = df['embedding'].apply(lambda x: np.dot(query_embedding.detach().numpy(), np.array(x)) / (np.linalg.norm(query_embedding.detach().numpy()) * np.linalg.norm(np.array(x))))

            top_indices = cosine_similarities.sort_values(ascending=False).head(5).index
            
            return_image_paths = df.loc[top_indices, 'file_name'].apply(lambda x: os.path.join('./coco_images_resized', x)).tolist()
            
            return return_image_paths
    
    if term and image_path:
        print("***TERM AND IMAGE PATH PROVIDED***")
        
        image = preprocess(Image.open(image_path)).unsqueeze(0)
        image_query = F.normalize(model.encode_image(image))
        text = tokenizer([term])
        text_query = F.normalize(model.encode_text(text))
        
        lam = 0.8
        
        query = F.normalize(lam * text_query + (1.0 - lam) * image_query)

        
        cosine_similarities = df['embedding'].apply(lambda x: np.dot(query.detach().numpy(), np.array(x)) / (np.linalg.norm(query.detach().numpy()) * np.linalg.norm(np.array(x))))

        top_indices = cosine_similarities.sort_values(ascending=False).head(5).index
        
        return_image_paths = df.loc[top_indices, 'file_name'].apply(lambda x: os.path.join('./coco_images_resized', x)).tolist()
        
        return return_image_paths

    if not term and not image_path:
        
        
        
        exit()


if __name__ == "__main__":
    perform_search("dog", None, False)