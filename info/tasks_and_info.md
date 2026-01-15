# Shelf env

<!-- ## Hooking book

### floating pos + yaw:

#### depth 1000 transformer
**run**: experiment_20251221_163123 **val_loss**:2.3e-4, **train_loss**:1e-5, **success**: %


#### dino cls 1000 transformer
**run**: experiment_20251221_164708 **val_loss**:1.2e-4, **train_loss**:1.2e-4, **success**: %


#### dino patch 1000 transformer
TODO maybe if time -->


### 7dof taskspace:

#### depth 1000 transformer

**run**: experiment_20251223_132104 **val_loss**:3.2e-3, **train_loss**:6.6e-4, **success**: 38% (100 evals) 

#### rgb 1000 transformer

**run**: experiment_20251225_174021 **val_loss**:2.7e-3, **train_loss**:6.2e-4, **success**: %

#### depth 1000 diffusion

**run**: experiment_20251228_232913 **val_loss**:3.8e-3, **train_loss**:2e-3, **success**: %



### 7dof jointspace:

#### minimax action normalization

**run**: experiment_20251229_152111 **val_loss**:2.6e-4, **train_loss**:1e-5, **success**: 45% (100 evals)

#### zscore action normalization

**run**: experiment_20251229_152849 **val_loss**:1.6e-3, **train_loss**:1.1e-4, **success**: 53% (100 evals)


### real data:

**run**: experiment_20260111_222201 **val_loss**:7e-4, **train_loss**:1e-5, **success**: 40% (50 evals)


# Table env

## Pushing cylinder:

### sim data:

#### depth 1000 resnet 18 trained from scratch transformer
**run**: experiment_20251222_161112 **val_loss**: 6e-5, **train_loss**: 1e-5 , **success**: 94% (100 evals)

#### rgb 1000 resnet 18 trained from scratch rgb
**run**: experiment_20260107_200958 **val_loss**: 9e-5, **train_loss**: 7e-5 , **success**: 99% (100 evals)


#### depth 1000 resnet 18 trained pretrained on imagenet transformer
**run**: experiment_20251222_184207 **val_loss**: 1e-4, **train_loss**: 1e-5 , **success**: TODO

#### dino cls depth 1000 transformer
**run**: experiment_20251222_171601 **val_loss**: 2e-4, **train_loss**: 7e-5 , **success**: TODO

#### dino patch 1000 resnet 18 trained from scratch transformer
TODO

#### depth 1000 resnet 18 trained from scratch diffusion

**run**: experiment_20260106_191304 **val_loss**: 2.8e-4, **train_loss**: 9.8e-5 , **success**: 60% (100 evals)




### real data:

#### lerobot diffusion policy
**loss**: 0.06 ? whatever that means

#### depth 1000 resnet 18 trained from scratch transformer
**run**: experiment_20251223_184825 **val_loss**: 3.7e-5, **train_loss**: 8e-6 , **success**: 62% success rate (62 from 100 tries) 

#### dino cls features
**run**: experiment_20251218_194527 **val_loss**: 7e-4, **train_loss**: 3e-4 , **success**: TODO


<!-- 
## pulling book?

TODO not really prio, extract experiments from ~october? -->