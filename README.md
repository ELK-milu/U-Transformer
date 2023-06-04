# U-Transformer

Source from [Diffusers](https://github.com/huggingface/diffusers)

These code is just for a custom simple unconditional-Traning.

use these dirs to replace same files in Diffusers

src/diffuser is used to replace the files of lib"diffusers" in local PIP（you should back up the "diffusers" before replace them)

The [Diffusers](https://github.com/huggingface/diffusers) is update frequently,these script and models is avaliable in the past month,but it may be unavaliable in the future（or now).

------

改自[Diffusers](https://github.com/huggingface/diffusers)

使用这些文件替换Diffusers库中的部分代码

这些代码用于一个自制的简单的无条件训练

src/diffuser内的文件也可用于替换本地PIP安装的diffusers（使用前最好备份）

[Diffusers](https://github.com/huggingface/diffusers)仓库更新速度很快，这些改动在一个月前可用，但是未来（或者现在）可能用不了

最近刚检查了[Diffusers](https://github.com/huggingface/diffusers)的新改动，发现他们也修改了对应的CrossAttenBlock2D部分

所以直接使用官方的CrossAttenBlock2D就行了
