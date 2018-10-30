# Git and GitHub

by Priya and Swati 

What is a Version Control System and why do we need it ? 
  * To keep track of the history of the file changes
  * To collaborate
  * To maintain a structure of the project
  * Track ownership
  * Track change evolution
 
The most common and useful type of version control system used is the distributed version control system. Git is the most famoust VCS out there. It is much faster than other DVCS like subversion.

**Github**
  * A website which provides hosting service for version control systems using git 
  * Also provides multiple other functionalities 
    * Visualization
    * Common user interface
    * Gist
    
**Available GUI applications for GitHub**
  * https://git-scm.com/download/gui/mac
  * https://git-scm.com/download/gui/windows
  * https://git-scm.com/download/gui/linux

**Commonly used terms**
  * Clone
    Create instance of the repository
    Checks out working copy and mirrors the entire repository
  * Pull
    Copy changes from remote repository to local repository
    Used for synchronization
  * Push
    Copy changes from local repository to remote repository
    Used to permanently update the changes into Git repository
  * Commits
    Hold the current state of the repository
  * Branches
    Can create another line of development (like a new feature)
    Default branch: Master
    New branch can be merged with Master upon completion
  * Revision
    Represents version of the source code (by commits)
  * Tag
    Assign a meaningful name with a specific version of the repository
  * Head
    Pointer to latest commit in the branch
  * Fork
    Create an independent copy of a repository from an existing one

**Git commands**
  * To clone existing repository:
    git clone https://github.com/swatirama/demo-repository.git
  * Start tracking new/edited files (add to staging area):
    git add filename
  * Start tracking all changed/new files:
    git add . (git add -A)
  * Save changes to local repository:
    git commit -m “Changes introduced in this commit”
  * Update remote repository:
    git push origin master
  * Pull from remote repository:
    git pull
  * Check status of local repository:
    git status
  * Get a log of all previous commits:
    git log
  * View staged and unstaged changes:
    git diff
  * Merge remote and your branch:
    git merge

**Additional Commands**
  * Remove untracked files:
    git clean
  * Reset HEAD to specified state (revert to older commit):
    git reset
  * Ignore automatically generated files from tracking:
    .gitignore file (update its contents)
  * Fetch from several repositories:
    git fetch




