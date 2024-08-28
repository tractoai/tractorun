# Changelog

## 0.16.0 (2024-08-28)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.15.0...0.16.0)

### Changes

- NYT-654: fix stderr reader stopping if yt operation is incorrect (#57). [1eb1c3a](https://github.com/tractoai/tractorun/commit/1eb1c3a5f07a526644c54b3585ab63449ea31072)
    

## 0.15.0 (2024-08-28)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.14.0...0.15.0)

### Changes

- NYT-653: fix stderr reader in case of process_per_node > 1 (#56). [590f396](https://github.com/tractoai/tractorun/commit/590f3961332819ed275adb7566200fb2943deee0)
    

## 0.14.0 (2024-08-26)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.13.0...0.14.0)

### Changes

- Create training dir with recursuve=True. [f1441bf](https://github.com/tractoai/tractorun/commit/f1441bf6703433491e5ab1162e74b8e349e91aee)
    
- Run tests on pull-requests (#38). [2be4551](https://github.com/tractoai/tractorun/commit/2be4551ff93ad3021cc167c8396b82b35a982a04)
    

## 0.13.0 (2024-08-21)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.12.0...0.13.0)

### Changes

- NYT-641: add tests for public interface, make public interface better (#52). [a2442cf](https://github.com/tractoai/tractorun/commit/a2442cf56c01a17363ba9eab1432214151eb58c9)
    
- NYT-641: hide extra imports in public code (#51). [2d86fd4](https://github.com/tractoai/tractorun/commit/2d86fd4f2f1dd8d6843ea154cb3bcc4641d8522c)
    
- [NYT-641] make stable public interface (#50). [36275ca](https://github.com/tractoai/tractorun/commit/36275ca691459ae932cae23014af6d09429be0ec)
    

## 0.12.0 (2024-08-20)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.11.0...0.12.0)

### Changes

- NYT-639: add yaml to dependencies (#49). [bf1afbc](https://github.com/tractoai/tractorun/commit/bf1afbc6914a486fd8bf134bfed091f6f71d40bf)
    
- Delete empty file tractorun/dataset.py (#48). [3151eff](https://github.com/tractoai/tractorun/commit/3151efffbf276183ab4f982445e9aaa57f88cb59)
    
- NYT-639: fix tractorun deps (#47). [2d29963](https://github.com/tractoai/tractorun/commit/2d29963cb30e99449c4890bfec7ec34f8b1e5986)
    
- NYT-629: stop stderr reader if operation fails (#46). [6413351](https://github.com/tractoai/tractorun/commit/6413351402be2c2d274afacca4eb6082c02414d5)
    
- NYT-629: add stderr read option (#45). [b1608c8](https://github.com/tractoai/tractorun/commit/b1608c809396f8955c7b584dd90db21c5e2f2ecb)
    
- NYT-629: move stderr to common codebase, add tests (#44). [5eededf](https://github.com/tractoai/tractorun/commit/5eededfcba2a548c9dcefbb576e2bb1597436b61)
    
- NYT-629: create base training dirs during local run (#43). [163c8f6](https://github.com/tractoai/tractorun/commit/163c8f684b9be3bd982af3b88b17007871dbb126)
    
  some preparations for stderr reader

## 0.11.0 (2024-08-14)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.10.0...0.11.0)

### Changes

- NYT-627: make bindings behave like docker/docker-compose (#42). [de9c66c](https://github.com/tractoai/tractorun/commit/de9c66c7a0db1d8036a225808ba011d28c164bb2)
    

## 0.10.0 (2024-08-13)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.9.0...0.10.0)

### Changes

- NYT-624: fix logging, terminate sidecars if operation is failed (#41). [60e74af](https://github.com/tractoai/tractorun/commit/60e74afe9b845700cf4bf6e040c33edb3df91b35)
    
- NYT-624: test infiniband on plack (#40). [4745613](https://github.com/tractoai/tractorun/commit/4745613b3e20b59b79645b94f0e84fc8b0d6dacf)
    

## 0.9.0 (2024-08-09)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.8.0...0.9.0)

### Changes

- Workaround vpc problems (#39). [01af5fe](https://github.com/tractoai/tractorun/commit/01af5feb1f7df43fa89653e2bf78a3ecfc50c98a)
    

## 0.8.0 (2024-08-08)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.7.0...0.8.0)

### Changes

- Update README.md (#37). [140c9fb](https://github.com/tractoai/tractorun/commit/140c9fbc986b9a053e48fde7dd23ad85d5efbc86)
    
- Add env to cli arg params and fix test's runner (#36). [dc78a85](https://github.com/tractoai/tractorun/commit/dc78a853c719f9b54b0b5bcc7eea51e953d41172)
    
- NYT-598: generalize run tests script (#35). [14af613](https://github.com/tractoai/tractorun/commit/14af6136abc594b194ae18e5d4ecc4c465d44c63)
    
- NYT-598: use yt_path fixture in all tests (#34). [dc2df95](https://github.com/tractoai/tractorun/commit/dc2df956ec026e929e1e408fdb80d5e8ff0d38da)
    
- NYT-598: add test for tensorproxy config (#33). [f8ed865](https://github.com/tractoai/tractorun/commit/f8ed865ec3aab7d15331c45115c0c6d8f2ca235a)
    
- Fix git tag in pypi action (#32). [27c1cf2](https://github.com/tractoai/tractorun/commit/27c1cf2ec03e353ae99c8bb136c346e477b27093)
    
- NYT-557: build and upload to pypi (#31). [4d7b2c2](https://github.com/tractoai/tractorun/commit/4d7b2c2d29891dbc34869f6ecb4254f86c6b52de)
    
- Add environment variables (#30). [76edbb7](https://github.com/tractoai/tractorun/commit/76edbb7f6724d1a747b8e6e44fd923434499593f)
    
- NYT-598: add tensorproxy magic sidecar (#29). [a59bab8](https://github.com/tractoai/tractorun/commit/a59bab8580e6a9a883e3ae96740a876c193eb8eb)
    
- NYT-598: rename bind args (#28). [9333240](https://github.com/tractoai/tractorun/commit/9333240b20cd187d83a7bf985053392048d4c097)
    
- NYT-598: add tensorproxy prototype (#27). [7b5bc53](https://github.com/tractoai/tractorun/commit/7b5bc53f6b19130451e60b51509429a341a8ad47)
    
- NYT-598: commit some preparation for tensorproxy (#26). [fa27dca](https://github.com/tractoai/tractorun/commit/fa27dca6205c901757ce7610e4788649655158a5)
    
- NYT-589: add sidecars (#25). [b931474](https://github.com/tractoai/tractorun/commit/b931474390186b99569c067e4d8bf4cfb61ec9ae)
    
- NYT-590: add lib bindings, use zip everywhere (#22). [71bd52f](https://github.com/tractoai/tractorun/commit/71bd52f31e69a9790ddd1ece56e11f5291b6e1bb)
    
- Add build-and-upload-a-wheel script (#23). [d1c3fec](https://github.com/tractoai/tractorun/commit/d1c3fec814ed091240067f8565bf108e2e50ca0b)
    
- Fix style: use attrs instead of attr, improve typing (#24). [93be372](https://github.com/tractoai/tractorun/commit/93be3724c547d362cb2b73e39c55ee36275f2333)
    
- Set version tag on the commit with version bump (#21). [f26d0ef](https://github.com/tractoai/tractorun/commit/f26d0ef64ae74434f52c49004483d60fb890de09)
    

## 0.7.0 (2024-07-15)

[Compare the full difference.](https://github.com/tractoai/tractorun/compare/0.6.0...0.7.0)

### Changes

- NYT-556: bump version via gh workflow (#19). [65be55c](https://github.com/tractoai/tractorun/commit/65be55cc675862a29fcb121c9213be90daabc4e7)
    
- Delete an empty file (#18). [68a83b7](https://github.com/tractoai/tractorun/commit/68a83b7d4c23f49d81a3aecc730be2ceeefafb7d)
    
- Add infiniband speed test (#16). [287e048](https://github.com/tractoai/tractorun/commit/287e048fa9e268eb80d91681ff78bc43f2d79e52)
    

## 0.6.0 (2024-07-12)
