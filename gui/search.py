from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import open_clip
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances 




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
        similarity_scores = cosine_similarities.loc[top_indices].tolist()
        return return_image_paths, similarity_scores
        
    if image_path and not term:
        print("***IMAGE PATH PROVIDED***")
        # Add logic to handle image search

        if use_pc:
            # TODO: Implement image search using principal components
            train_images, train_image_names = load_images('./coco_images_resized', max_images=2000)
            
            k = use_pc
            pca = PCA(n_components=k)
            pca.fit(train_images)
            
            transform_images, transform_image_names = load_images('./coco_images_resized', max_images=10000, target_size=(224, 224))
            reduced_embeddings = pca.transform(transform_images)
            
            query_embedding = pca.transform(query_embedding.detach().numpy().reshape(1, -1))
            nearest_indices, distances = nearest_neighbors(query_embedding, reduced_embeddings, top_k=5)
            
            return_image_paths = [os.path.join('./coco_images_resized', transform_image_names[idx]) for idx in nearest_indices]
            similarity_scores = distances.tolist()
            
            return return_image_paths, similarity_scores
            
        else:
                                
            image = preprocess(Image.open(image_path)).unsqueeze(0)
            
            query_embedding = F.normalize(model.encode_image(image))
                        
            cosine_similarities = df['embedding'].apply(lambda x: np.dot(query_embedding.detach().numpy(), np.array(x)) / (np.linalg.norm(query_embedding.detach().numpy()) * np.linalg.norm(np.array(x))))

            top_indices = cosine_similarities.sort_values(ascending=False).head(5).index
            
            return_image_paths = df.loc[top_indices, 'file_name'].apply(lambda x: os.path.join('./coco_images_resized', x)).tolist()
            
            similarity_scores = cosine_similarities.loc[top_indices].tolist()
            return return_image_paths, similarity_scores
    
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
        
        similarity_scores = cosine_similarities.loc[top_indices].tolist()
        return return_image_paths, similarity_scores

    if not term and not image_path:
        
        
        
        exit()

def nearest_neighbors(query_embedding, embeddings, top_k=5):
    # query_embedding: The embedding of the query item (e.g., the query image) in the same dimensional space as the other embeddings.
    # embeddings: The dataset of embeddings that you want to search through for the nearest neighbors.
    # top_k: The number of most similar items (nearest neighbors) to return from the dataset.
    # Hint: flatten the "distances" array for convenience because its size would be (1,N)
    distances = euclidean_distances(query_embedding.reshape(1, -1), embeddings).flatten()
    nearest_indices = np.argsort(distances)[:top_k]
    return nearest_indices, distances[nearest_indices]

def load_images(image_dir, max_images=None, target_size=(224, 224)):
    images = []
    image_names = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            img = img.convert('L')  # Convert to grayscale ('L' mode)
            img = img.resize(target_size)  # Resize to target size
            img_array = np.asarray(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array.flatten())  # Flatten to 1D
            image_names.append(filename)
        if max_images and i + 1 >= max_images:
            break
    return np.array(images), image_names

if __name__ == "__main__":
    perform_search("dog", None, False)