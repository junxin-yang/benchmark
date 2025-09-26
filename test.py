import matplotlib.font_manager
for f in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
    print(f)