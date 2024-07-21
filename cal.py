def determine_comm_schedule(comm_dict, bp_dict, H):
    layers = list(comm_dict.keys())
    num_layers = len(layers)
    total_bp_time = sum(bp_dict.values())

    # dp_cache stores the tuple (minimum uncovered time, min iterations, schedule)
    dp_cache = {}

    def find_optimal_comm(index, accumulated_bp_time):
        if index >= num_layers:
            return (0, 0, {})  # Base case: no more layers to process

        if (index, accumulated_bp_time) in dp_cache:
            return dp_cache[(index, accumulated_bp_time)]

        # Initialize optimal values for uncovered time, iteration count and schedule
        min_uncovered_time = float('inf')
        min_iterations = float('inf')
        optimal_schedule = {}

        for i in range(index, num_layers):
            current_comm_time = sum(comm_dict[layers[j]] for j in range(index, i + 1))
            next_bp_time = accumulated_bp_time + bp_dict[layers[i]]
            
            # Calculate uncovered communication time if this layer is the last communicated in this iteration
            uncovered_time = max(0, current_comm_time - accumulated_bp_time)

            # Recursive call for next layers
            next_result = find_optimal_comm(i + 1, next_bp_time)
            next_uncovered_time, next_iterations, next_schedule = next_result

            total_uncovered_time = uncovered_time + next_uncovered_time

            # Check if the current choice is better (minimize uncovered time, then iterations)
            if (total_uncovered_time < min_uncovered_time or
                (total_uncovered_time == min_uncovered_time and next_iterations + 1 < min_iterations)):
                min_uncovered_time = total_uncovered_time
                min_iterations = next_iterations + 1
                optimal_schedule = next_schedule.copy()
                for j in range(index, i + 1):
                    optimal_schedule[layers[j]] = min_iterations  # Store iteration number for these layers

        dp_cache[(index, accumulated_bp_time)] = (min_uncovered_time, min_iterations, optimal_schedule)
        return dp_cache[(index, accumulated_bp_time)]

    # Start the recursion from the first layer and no accumulated backward time
    total_uncovered_time, total_iterations, optimal_schedule = find_optimal_comm(0, 0)
    return optimal_schedule, total_iterations, total_uncovered_time  # Return the schedule, iteration count, and uncovered time

# Example usage:
comm_dict = {
    "conv1": 0.026038408279418945,
    "bn1": 0.0006838526044573103,
    "layer1.0.conv1": 0.002113001687186105,
    "layer1.0.bn1": 0.0007846355438232422,
    "layer1.0.conv2": 0.0020205633980887277,
    "layer1.0.bn2": 0.0008171626499720982,
    "layer1.0.shortcut": 2.288818359375e-05,
    "layer1.1.conv1": 0.00199716431753976,
    "layer1.1.bn1": 0.0008359977177211217,
    "layer1.1.conv2": 0.002042055130004883,
    "layer1.1.bn2": 0.0008117812020438058,
    "layer1.1.shortcut": 1.6450881958007812e-05,
    "layer2.0.conv1": 0.003943783896309989,
    "layer2.0.bn1": 0.0011701243264334543,
    "layer2.0.conv2": 0.008086136409214564,
    "layer2.0.bn2": 0.0017409324645996094,
    "layer2.0.shortcut.0": 0.0007473400660923549,
    "layer2.0.shortcut.1": 0.0005331720624651228,
    "layer2.1.conv1": 0.007887772151402064,
    "layer2.1.bn1": 0.001825230462210519,
    "layer2.1.conv2": 0.008019345147269112,
    "layer2.1.bn2": 0.0015464510236467635,
    "layer2.1.shortcut": 1.5837805611746652e-05,
    "layer3.0.conv1": 0.015847410474504744,
    "layer3.0.bn1": 0.0029815265110560824,
    "layer3.0.conv2": 0.031623125076293945,
    "layer3.0.bn2": 0.005389281681605748,
    "layer3.0.shortcut.0": 0.0018948146275111607,
    "layer3.0.shortcut.1": 0.0007279259817940849,
    "layer3.1.conv1": 0.03149448122297015,
    "layer3.1.bn1": 0.005289861134120396,
    "layer3.1.conv2": 0.03129638944353376,
    "layer3.1.bn2": 0.0058983394077845985,
    "layer3.1.shortcut": 1.655306134905134e-05,
    "layer4.0.conv1": 0.04206265722002302,
    "layer4.0.bn1": 0.00133340699332101,
    "layer4.0.conv2": 0.08136272430419922,
    "layer4.0.bn2": 0.001160928181239537,
    "layer4.0.shortcut.0": 0.007226603371756417,
    "layer4.0.shortcut.1": 0.0014796257019042969,
    "layer4.1.conv1": 0.0812403815133231,
    "layer4.1.bn1": 0.0016089848109654018,
    "layer4.1.conv2": 0.08116851534162249,
    "layer4.1.bn2": 0.0014511517116001674,
    "layer4.1.shortcut": 1.4441353934151785e-05,
    "linear": 0.0010708740779331752
}
bp_dict = {
    "conv1": 3.473612726951132e-05,
    "bn1": 0.000709854826635244,
    "layer1.0.conv1": 0.001968982268352898,
    "layer1.0.bn1": 0.0005305153982979911,
    "layer1.0.conv2": 0.0019675517568782885,
    "layer1.0.bn2": 0.0003737430183254943,
    "layer1.0.shortcut": 0.0003938188358229034,
    "layer1.1.conv1": 0.0019634840439776984,
    "layer1.1.bn1": 0.0005319702381990394,
    "layer1.1.conv2": 0.0020079807359345107,
    "layer1.1.bn2": 0.00037572335223762357,
    "layer1.1.shortcut": 0.00039345390942631934,
    "layer2.0.conv1": 0.001704074898544623,
    "layer2.0.bn1": 0.0002826671211086974,
    "layer2.0.conv2": 0.0010443512274294483,
    "layer2.0.bn2": 0.00021218280402981505,
    "layer2.0.shortcut.0": 0.0013259235693483936,
    "layer2.0.shortcut.1": 0.00036855133212342555,
    "layer2.1.conv1": 0.001047669624795719,
    "layer2.1.bn1": 0.0002834796905517578,
    "layer2.1.conv2": 0.001089748071164501,
    "layer2.1.bn2": 0.00022623490314094388,
    "layer2.1.shortcut": 0.00021787079013123805,
    "layer3.0.conv1": 0.0013505196084781569,
    "layer3.0.bn1": 0.00022077560424804688,
    "layer3.0.conv2": 0.0007617814200265067,
    "layer3.0.bn2": 0.00017934429402254064,
    "layer3.0.shortcut.0": 0.0010346587823361767,
    "layer3.0.shortcut.1": 0.00027266327215700735,
    "layer3.1.conv1": 0.0007636401118064413,
    "layer3.1.bn1": 0.00023721188915019133,
    "layer3.1.conv2": 0.0008228117105912189,
    "layer3.1.bn2": 0.00018656010530432876,
    "layer3.1.shortcut": 0.0001706590457838409,
    "layer4.0.conv1": 0.00328648820215342,
    "layer4.0.bn1": 0.00023343125168158083,
    "layer4.0.conv2": 0.000858306884765625,
    "layer4.0.bn2": 0.00018026877422722018,
    "layer4.0.shortcut.0": 0.0009318273894640864,
    "layer4.0.shortcut.1": 0.0002704639824069276,
    "layer4.1.conv1": 0.0008513927459716797,
    "layer4.1.bn1": 0.00022442000252859934,
    "layer4.1.conv2": 0.0014063582128408005,
    "layer4.1.bn2": 0.00042083312054069674,
    "layer4.1.shortcut": 0.00045894116771464445,
    "linear": 0.0011746250853246572
}

schedule, total_iterations, total_uncovered_time = determine_comm_schedule(comm_dict, bp_dict, H=10)
print("Schedule:", schedule)
print("Total Iterations:", total_iterations)
print("Total Uncovered Communication Time:", total_uncovered_time)
