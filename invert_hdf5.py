import argparse
import h5py
from tqdm import tqdm
from inversion.ddim_inversion import DDIMInversionModel
from inversion.ddpm_inversion import DDPMInversionModel


def main(args):
	original_file = args.original_file
	new_file_name = args.new_file
	hdf5_file = h5py.File(original_file, 'r')


	image_size = args.image_size
	num_trajs = args.num_trajs
	obs_key = args.obs_key

	num_steps, inversion_steps = args.num_steps, args.inversion_steps
	version = args.version
	
	if args.model == 'ddim':
		inversion_model = DDIMInversionModel(num_steps=num_steps, inversion_steps=inversion_steps, img_size=(image_size, image_size))
	elif args.model == 'ddpm':
		inversion_model = DDPMInversionModel(num_steps=num_steps, inversion_steps=inversion_steps, config=None, version=version)
	else:
		raise ValueError(f"Invalid model {args.model}")
	
	
	batch_size = args.batch_size

	new_file = h5py.File(new_file_name, 'w')
	new_data_group = new_file.create_group('data')
	traj_names = [f'demo_{i}' for i in range(num_trajs)]
	for demo_key in tqdm(hdf5_file['data'].keys()):
		if demo_key not in traj_names:
			continue
		hdf5_file.copy(hdf5_file['data'][demo_key], new_data_group, name=demo_key)
		obs = hdf5_file['data'][demo_key]['obs']

		new_obs = obs[obs_key][:]
		if inversion_model is not None:
			for i in range(0, len(new_obs), batch_size):
				new_obs[i:i+batch_size] = inversion_model.invert(new_obs[i:i+batch_size])
		
		if new_file["data"][demo_key]["obs"][obs_key].shape[1:3] != (image_size, image_size):
			del new_file["data"][demo_key]["obs"][obs_key]
			new_file["data"][demo_key]["obs"][obs_key] = new_obs
		else:
			new_file["data"][demo_key]["obs"][obs_key][:] = new_obs
	

	hdf5_file.close()
	new_file.close()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--original_file", type=str, default='datasets/demo.hdf5')
	parser.add_argument("--new_file", type=str, default='datasets/test.hdf5')
	parser.add_argument("--image_size", type=int, default=360)
	parser.add_argument("--num_trajs", type=int, default=100)
	parser.add_argument("--obs_key", type=str, default='base_camera')
	parser.add_argument("--num_steps", type=int, default=50)
	parser.add_argument("--inversion_steps", type=int, default=5)
	parser.add_argument("--version", type=str, default='2.1')
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument('--model', type=str, default='ddpm', choices=['ddpm', 'ddim'])
	
	args = parser.parse_args()
	main(args)