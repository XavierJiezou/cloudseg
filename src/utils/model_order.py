model_order = dict(
    cloudsen12_high_l1c=["unetmobv2","hrcloudnet","cdnetv2","dbnet","cdnetv1","mcdnet","kappamask","scnn"],
    cloudsen12_high_l2a=["unetmobv2","hrcloudnet","cdnetv2","dbnet","cdnetv1","kappamask","mcdnet","scnn"],
    gf12ms_whu_gf1=["unetmobv2","hrcloudnet","kappamask","cdnetv2","dbnet","cdnetv1","mcdnet","scnn"],
    gf12ms_whu_gf2=["hrcloudnet","unetmobv2","dbnet","mcdnet","cdnetv2","cdnetv1","scnn","kappamask"],
    hrc_whu=["hrcloudnet","unetmobv2","cdnetv1","dbnet","cdnetv2","kappamask","scnn","mcdnet"],
    l8_biome_crop=["kappamask","hrcloudnet","unetmobv2","cdnetv2","dbnet","cdnetv1","scnn","mcdnet"],
)