# Model weight

The final stage-2 checkpoint is:

```text
weights/best.pth
```

It is a PyTorch `state_dict` for the released artifact model
(`image_size=256`, `style_dim=512`, `feature_dim=40`, `motion_dim=40`,
`pure_data=False`) and is tracked with Git LFS. Verify it with:

```bash
shasum -a 256 -c weights/SHA256SUMS
```
