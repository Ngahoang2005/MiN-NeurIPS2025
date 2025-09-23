def get_model(args, loger):
    name = args['model']

    if name == 'min':
        from models.MiN import MinNet
        return MinNet(args, loger)
    else:
        assert "No such model!"
