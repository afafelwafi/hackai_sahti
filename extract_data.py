import os
import requests
import zipfile
from config_loader import get_config


def download_github_repo(repo_url: str = None, extract_to: str = None) -> str:
    """
    Download and extract GitHub repository as ZIP.
    
    Args:
        repo_url (str): GitHub repo URL. If None, uses config default.
        extract_to (str): Where to extract files. If None, uses config default.
        
    Returns:
        str: Path to extracted repository folder
    """
    config = get_config()
    
    # Use parameters or config defaults
    if repo_url is None:
        repo_url = config.repo_url
    if extract_to is None:
        extract_to = config.dataset_folder
    
    # Get repo name from URL
    repo_name = repo_url.split('/')[-1]
    extract_to = os.path.join(os.getcwd(), extract_to)
    
    # Download URL
    zip_url = f"{repo_url}/archive/refs/heads/main.zip"
    zip_file = f"{extract_to}.zip"
    
    print(f"Downloading {repo_name}...")
    
    try:
        # Download
        response = requests.get(zip_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(zip_file, 'wb') as f:
            f.write(response.content)
        
        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Cleanup
        os.remove(zip_file)
        
        extracted_folder = os.path.join(extract_to, f"{repo_name}-main")
        print(f"Done! Files extracted to: {extracted_folder}")
        
        return extracted_folder
        
    except requests.RequestException as e:
        print(f"Error downloading repository: {e}")
        raise
    except zipfile.BadZipFile as e:
        print(f"Error extracting ZIP file: {e}")
        if os.path.exists(zip_file):
            os.remove(zip_file)
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        if os.path.exists(zip_file):
            os.remove(zip_file)
        raise


# Example usage for the BMD-HS-Dataset
if __name__ == "__main__":
    # Download the BMD-HS-Dataset using config defaults
    folder = download_github_repo()
    print(f"Dataset downloaded and extracted to: {folder}")