1. Clone papyrax git@github.com:nebius/papyrax.git
2. Patch papyrax:
   ```
   On branch main
   Your branch is up to date with 'origin/main'.

   Changes not staged for commit:
   (use "git add <file>..." to update what will be committed)
   (use "git restore <file>..." to discard changes in working directory)
   modified:   papyrax/utils/tqdm.py

   Untracked files:
   (use "git add <file>..." to include in what will be committed)
    pyrax/py.typed

   no changes added to commit (use "git add" and/or "git commit -a")
   ```
   
   ```
    diff --git a/papyrax/utils/tqdm.py b/papyrax/utils/tqdm.py
    index de34eb53..fe400300 100644
    --- a/papyrax/utils/tqdm.py
    +++ b/papyrax/utils/tqdm.py
    @@ -42,7 +42,6 @@ def logging_redirect_tqdm(
         loggers: list[logging.Logger] | None = None,
         tqdm_class: Type = std_tqdm,
     ) -> Iterator[None]:
    -    # type: (...) -> Iterator[None]
         """
         A fixed version tqdm.contrib.logging.logging_redirect_tqdm.
   ```
3. Install papyrax `pip install -r requirements.txt` `SETUPTOOLS_ENABLE_FEATURES="legacy-editable" python -m pip install -e .`. SETUPTOOLS_ENABLE_FEATURES is important
4. `make all-all-check`
5. Install extra deps: https://github.com/nebius/papyrax/blob/05dc337efafecb816484ac9bd930943dd863cad6/docker/Dockerfile#L18
