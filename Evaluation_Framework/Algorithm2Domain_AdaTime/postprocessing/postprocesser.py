import json 
def postprocess(input_data):
    # read json
    with open("./config/postprocessing_config.json", 'r') as file:
        data = json.load(file)

    # apply postprocessing steps
    for step in data.get("postprocessing_steps", []):
        output_channel = step.get("output_channel", 0)
        step_type = step.get("steps", {}).get("type", "")
        step_params = step.get("steps", {}).get("params", {})

        # input data is an array of shape [batch_size, num_channels, predictions]
        channel_data = input_data[:, output_channel]

        if step_type == "interval_to_label":
            intervals = step_params.get("intervals", [])
            for pred in channel_data:
                for interval in intervals:
                    if interval["start"] <= pred < interval["end"]:
                        pred = interval["label"]
                        break
                    
    return input_data