from Yenos_v2 import Model, Preprocessor


model = Model.Model()
preprocessor = Preprocessor.Preprocessor()


name_data = "Antoinette"

process_data = preprocessor.process_name(name=name_data)

predictions = model.predict(process_data)

print(predictions)

