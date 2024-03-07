from git import Repo
from github_creds import github_repo_url, github_username, github_token

def push_to_github(local_repo_path, github_repo_url, github_username, github_token):
    repo = Repo(local_repo_path)

    try:
        # Add dataset.csv to the index
        repo.index.add("dataset.csv")

        # Commit changes
        repo.index.commit("Automated commit - updated dataset.csv")

        # Set up remote repository
        origin = repo.remote(name='origin')
        origin_url = origin.config_reader.get("url")

        # Push changes to GitHub
        origin.push()

        print("Pushed to GitHub successfully.")
    except Exception as e:
        print("Error:", e)

local_repo_path = "./"

push_to_github(local_repo_path, github_repo_url, github_username, github_token)
