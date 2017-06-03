require "ability_data"
require "nn"

-- Hyper parameters
local MINI_BATCH_SIZE = 100 -- number of examples in a batch
local PATIENCE = 15 -- how long we should put up with the validation error increasing before stopping
local EARLY_STOP_THRESHOLD = 0 -- difference between the previous and current validation error (0 = any time the error increases)
local HIDDEN_LAYERS = 3
local LEARNING_RATE = .1
local TRAINING_SET_SIZE = .8 -- training/test data split (training 80%, test 20%)

local function CreateContainer(input_layer, output_layer, hidden_layer)
	local net = nn.Sequential()

	net:add(nn.Linear(input_layer, hidden_layer))
	
	for i = 1, HIDDEN_LAYERS - 1 do
		net:add(nn.RReLU())
		net:add(nn.Linear(hidden_layer, hidden_layer))
	end
	
	net:add(nn.RReLU())
	net:add(nn.Linear(hidden_layer, output_layer))
	
	return net
end

local function Shuffle(data)
	for i = 1, #data do
		local j = torch.random(1, i)

		local temp = data[i]
		data[i] = data[j]
		data[j] = temp
	end
end

-- Label weights represents the weights of each criterion (so moving vs target vs items vs abilities), 
-- class weights represents the weights of each class in the classification criterions (so specific items within the error for items)
local function Loss(label_weights, class_weights)
	local loss = nn.ParallelCriterion()

	for i, label_weight in ipairs(label_weights) do
		if i == 1 then -- first part is the move data
			loss:add(nn.AbsCriterion(), label_weight)
		else
			local nll = nn.CrossEntropyCriterion(class_weights[i - 1])
			nll.nll.ignoreIndex = 0

			loss:add(nll, label_weight)
		end	
	end

	return loss
end

local function Train(net, data, loss, label_sizes)
	Shuffle(data) -- for randomizing the split

	local validation_split = math.floor(#data * TRAINING_SET_SIZE)

	local best = math.huge -- previous best validation error
	local training_err = 0
	local validation_err = 0
	local patience_itr = 0 -- how long we've been waiting for validation error to go back down

	print(net)

	while patience_itr < PATIENCE do
		print(string.format("training error %f, validation error %f, best %f, patience %d/%d", training_err, validation_err, best, patience_itr, PATIENCE))

		training_err = 0
		validation_err = 0

		training = torch.randperm(validation_split) -- shuffle training examples

		-- train
		for i = 1, training:size(1) do
			local example = data[training[i]]

			net:zeroGradParameters()

			local output = net:forward(example[1])

			local parts = {}
			local start = 1

			for index, size in ipairs(label_sizes) do -- split up the output into move data/classes...
				parts[index] = output:sub(1, -1, start, start + size - 1)
				start = start + size
			end

			training_err = training_err + loss:forward(parts, example[2])
			net:backward(example[1], torch.cat(loss:backward(parts, example[2])))

			net:updateParameters(LEARNING_RATE)
		end

		-- calculate validation error
		for i = validation_split + 1, #data do
			local example = data[i]

			local output = net:forward(example[1])

			local parts = {}
			local start = 1

			for index, size in ipairs(label_sizes) do
				parts[index] = output:sub(1, -1, start, start + size - 1)
				start = start + size
			end

			validation_err = validation_err + loss:forward(parts, example[2])
		end

		if best - validation_err < EARLY_STOP_THRESHOLD then
			patience_itr = patience_itr + 1
		else -- improvement was made
			patience_itr = 0
			best = validation_err
		end	
	end
end

local function ParseMoveBatch(examples, hero, team, totals)
	local batch_pos = 1
	local input_batch = {}
	local output_batch = {}
	local more = false

	local num_items = #ability_data.items[hero][team]
	local num_active_items = #ability_data.activeItems[hero][team]
	local num_active_abilities = #ability_data.activeAbilities[hero][team]

	local input_view = 0 -- counter for size of the input feature vector, so that we can shape the input batch appropriately later

	for example in examples do
		if batch_pos > MINI_BATCH_SIZE then -- we've filled a batch, but there's still more examples left to read
			more = true
			break
		end

		local chunk_pos = 1
		local state = 0
		local items_pos

		local input = {}
		local output = {}

		local move_info = {}

		for part in example:gmatch("[^,]+") do
			if state == 0 then -- state 0: input
				if part == "items" then
					state = 1
					items_pos = chunk_pos
					chunk_pos = 1

					-- pre-fill current items with zeroes
					for i = items_pos, items_pos + num_items - 1 do
						input[i] = 0.0
					end
				else
					input[chunk_pos] = tonumber(part)
					chunk_pos = chunk_pos + 1
				end
			elseif state == 1 then -- state 1: items
				if part == "output" then
					state = 2
				else
					input[(tonumber(part) - 1) + items_pos] = 1.0 -- we have that item, set it to 1
				end
			elseif state == 2 then -- state 2: move location
				if chunk_pos > 3 then -- beginning of classes
					state = 3
					local class = tonumber(part)

					-- Increment count of occurences of this class
					if totals[1] == nil then
						totals[1] = {}
					end

					if totals[1][class] == nil then
						-- Make sure the array doesn't have any holes in it
						for i = #totals[1] + 1, class - 1 do
							totals[1][i] = 0
						end

						totals[1][class] = 1
					else	
						totals[1][class] = totals[1][class] + 1
					end	

					output[1] = torch.Tensor(move_info)
					output[2] = tonumber(part)

					chunk_pos = 3
				else
					local move = tonumber(part)

					move_info[chunk_pos] = move
					chunk_pos = chunk_pos + 1
				end
			elseif state == 3 then -- state 3: classification output
				local class = tonumber(part)

				if totals[chunk_pos - 1] == nil then
					totals[chunk_pos - 1] = {}
				end

				if totals[chunk_pos - 1][class] == nil then
					for i = #totals[chunk_pos - 1] + 1, class - 1 do
						totals[chunk_pos - 1][i] = 0
					end

					totals[chunk_pos - 1][class] = 1
				else	
					totals[chunk_pos - 1][class] = totals[chunk_pos - 1][class] + 1
				end	

				output[chunk_pos] = class

				chunk_pos = chunk_pos + 1
			end
		end

		input_view = #input
		input_batch[batch_pos] = torch.Tensor(input)

		for i = 1, #output do
			if output_batch[i] == nil then
				output_batch[i] = {}
			end

			output_batch[i][batch_pos] = output[i]
		end	

		batch_pos = batch_pos + 1
	end

	if batch_pos > 1 then -- did we actually get any examples
		-- turn the batches into Tensors
		output_batch[1] = torch.view(torch.cat(output_batch[1]), -1, batch_pos - 1)
	
		for i = 2, #output_batch do
			output_batch[i] = torch.Tensor(output_batch[i])
		end	

		return {torch.view(torch.cat(input_batch), -1, input_view), output_batch}, more
	else
		return nil, more
	end
end

local function LoadData(hero, team)
	local path = string.format("data/%s/%d_", hero, team)

	local move_data = {} -- table of example batches
	local move_pos = 0 -- current position in the table
	local move_total = 0 -- total number of examples (not batches)

	--local items_data = {}
	--local items_pos = 0

	local move_class_counts = {}

	local move_lines = io.lines(path .. "moveexamples")
	local more = true
	
	while more do
		local batch
		
		batch, more = ParseMoveBatch(move_lines, hero, team, move_class_counts)

		if batch ~= nil then
			move_data[move_pos] = batch
			move_pos = move_pos + 1
			move_total = move_total + batch[1]:size(1)
		end	
	end
	
	--for example in io.lines(path .. "itemsexamples") do
	--	item_data, item_pos, {})

	--	item_pos = item_pos + 1
	--end

	-- Calculate weights for the loss function
	local move_label_weights = {1} -- weights of each label (move vs target vs abilities vs items...)
	local move_class_weights = {} -- weights of each class in the labels (specific items or abilities within that)

	for i, label in ipairs(move_class_counts) do
		local weights = {}

		for j, total in ipairs(label) do
			weights[j] = move_total / total -- = number of examples / number of examples with that class in the label
		end

		move_class_weights[i] = torch.Tensor(weights)
		move_label_weights[i + 1] = (label[0] or label[1]) / move_total -- number of examples where the label wasn't active / number of examples where the label was active
	end

	return move_data, items_data, move_label_weights, move_class_weights
end

for hero in paths.iterdirs("data") do
	print("Training " .. hero)
	paths.mkdir("data/" .. hero .. "/nets")

	do
		print("\nRadiant")

		local move_data, items_data, move_label_weights, move_class_weights = LoadData(hero, 2)

		if #move_data == 0 then
			print("Missing training data\n")
		else
			local input_len = move_data[1][1]:size(2)
			local output_len = 13 + #ability_data.activeAbilities[hero][2] + #ability_data.activeItems[hero][2]

			local move = CreateContainer(input_len, output_len, math.floor((input_len + output_len) / 2))

			print("\nMoving:")
			Train(move, move_data, Loss(move_label_weights, move_class_weights), {3, 8, #ability_data.activeAbilities[hero][2] + 1, #ability_data.activeItems[hero][2] + 1})
			torch.save("data/" .. hero .. "/nets/2_move", move, "ascii")

			--print("Items/build:")
			--Train(items, items_data, .1, nn.ClassNLLCriterion(), items_size)
			--torch.save(move, "../data" .. hero .. "2_itemsnn")
		end
	end

	do
		print("\nDire")

		local move_data, items_data, move_label_weights, move_class_weights = LoadData(hero, 3)

		if #move_data == 0 then
			print("Missing training data\n")
		else
			local input_len = move_data[1][1]:size(2)
			local output_len = 13 + #ability_data.activeAbilities[hero][3] + #ability_data.activeItems[hero][3]

			local move = CreateContainer(input_len, output_len, math.floor((input_len + output_len) / 2))

			print("\nMoving:")
			Train(move, move_data, Loss(move_label_weights, move_class_weights), {3, 8, #ability_data.activeAbilities[hero][3] + 1, #ability_data.activeItems[hero][3] + 1})
			torch.save("data/" .. hero .. "/nets/3_move", move, "ascii")

			--print("Items/build:")
			--Train(items, items_data, .1, nn.ClassNLLCriterion(), items_size)
			--torch.save(move, "../data" .. hero .. "3_itemsnn")
		end
	end
end
