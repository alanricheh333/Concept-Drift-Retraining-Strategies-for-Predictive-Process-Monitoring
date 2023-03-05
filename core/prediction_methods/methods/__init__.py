from .method import Method


def get_prediction_method(method_name):
    if method_name == "SDL":
        from methods.SDL import sdl
        return Method("SDL", sdl.train, sdl.test, sdl.update, {"epochs": 200, "early_stop": 10})
    # elif method_name == "DBN":
    #     from Methods.EDBN.Train import train, update
    #     from Methods.EDBN.Predictions import test
    #     return Method("DBN", train, test, update)

    # elif method_name == "CAMARGO":
    #     from Methods.Camargo import adapter as camargo
    #     return Method("Camargo", camargo.train, camargo.test, camargo.update, {"epochs": 200, "early_stop": 10})

    # elif method_name == "DIMAURO":
    #     from Methods.DiMauro import adapter as dimauro
    #     return Method("Di Mauro", dimauro.train, dimauro.test, dimauro.update, {"epochs": 200, "early_stop": 10})

    elif method_name == "LIN":
        from .Lin import adapter as lin
        return Method("Lin", lin.train, lin.test, lin.update, {"epochs": 200, "early_stop": 10})
    
    elif method_name == "PASQUADIBISCEGLIE":
        from .Pasquadibisceglie import adapter as pasquadibisceglie
        return Method("Pasquadibisceglie", pasquadibisceglie.train, pasquadibisceglie.test,
                      pasquadibisceglie.update, {"epochs": 200, "early_stop": 10})
    
    # elif method_name == "TAX":
    #     from Methods.Tax import adapter as tax
    #     return Method("Tax", tax.train, tax.test, tax.update, {"epochs": 200, "early_stop": 10})

    # elif method_name == "TAYMOURI":
    #     from Methods.Taymouri import adapter as taymouri
    #     return Method("Taymouri", taymouri.train, taymouri.test, {"epoch": 10})
    else:
        print("ERROR: method name not found!")
        print("ERROR: Possible methods are:" + ",".join(ALL))
        raise NotImplementedError()


ALL = ["SDL", "DBN", "CAMARGO", "DIMAURO", "LIN", "PASQUADIBISCEGLIE", "TAX", "TAYMOURI"]
