
class ImageClfExample:
  def __init__(self, example_id, image, label, other_features=None):
    """Image classification training example

    :param example_id: Unique example id
    :param image: Could be a image ids, PIL Image, or numpy array
    :param label: integer label
    :param other_features: dictionary of array, addition feature attached to this example
    """
    self.example_id = example_id
    self.image = image
    self.label = label
    self.other_features = other_features
