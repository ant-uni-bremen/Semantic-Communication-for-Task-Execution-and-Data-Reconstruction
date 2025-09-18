import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tikzplotlib

sns.set_theme(style="whitegrid", palette="colorblind")
def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

class SemanticPlots:
    def __init__(self, models_to_use,split_image_intos,enc_out_dec_inps,train_snrs,alphas,betas,num_images,SNR_range,loss_functions,use_pgf):
        self.alphas = alphas
        self.betas = betas
        self.models_to_use = models_to_use
        self.split_image_intos = split_image_intos
        self.enc_out_dec_inps = enc_out_dec_inps
        self.train_snrs = train_snrs
        self.num_images = num_images
        self.SNR_range = SNR_range
        self.loss_functions = loss_functions
        self.plot_marker = {
            'CNN': 'o',        # circle
            'ResNet14': 's',   # square
            'ResNet20': 'D',   # diamond
            'VIT': '^',        # triangle_up
        }

        #configure(use_pgf)
    
    def psnr(self,value,folderpath,fig_size=(20,12)):
        plt.figure(figsize=fig_size)
        for lossfu in self.loss_functions:
            for beta in self.betas:
                for alpha in self.alphas:
                    for enc_out_dec_inp in self.enc_out_dec_inps:
                        for train_snr in self.train_snrs:
                            for splits in self.split_image_intos:
                                for model in self.models_to_use:
                                    psnr = (value.get(f'{lossfu}',{})
                                                .get(f'{beta}',{})
                                                .get(f'{alpha}', {})
                                                .get(f'{enc_out_dec_inp}', {})
                                                .get(f'{train_snr}', {})
                                                .get(f'{splits}', {})
                                                .get(model, {})
                                                .get("psnr"))
                                    plt.plot(self.SNR_range,np.squeeze(psnr),marker=self.plot_marker.get(f'{model}',{}),ms=10,label=f"{model}/loss Fnc:{lossfu}/beta:{beta}/alpha:{alpha}/bn:{enc_out_dec_inp}/Train SNR:{train_snr}/splits:{splits}")
        plt.xlabel('SNR[dB]', fontsize=16, fontweight='bold')
        plt.ylabel('PSNR [dB]', fontsize=16, fontweight='bold')
        plt.grid(True)
        plt.xticks(fontsize=16, fontweight='bold')  
        plt.yticks(fontsize=16, fontweight='bold')  
        # move legend out of the plot
        #plt.legend(prop={'size': 16, 'weight': 'bold'})
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folderpath, f'psnr.png'))
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(os.path.join(folderpath, f'psnr.tex'))
        plt.close()

    def ssim(self,value,folderpath,fig_size=(20,12)):
        plt.figure(figsize=fig_size)
        for lossfu in self.loss_functions:
            for beta in self.betas:
                for alpha in self.alphas:
                    for enc_out_dec_inp in self.enc_out_dec_inps:
                        for train_snr in self.train_snrs:
                            for splits in self.split_image_intos:
                                for model in self.models_to_use:
                                    ssim = (value.get(f'{lossfu}',{})
                                                .get(f'{beta}',{})
                                                .get(f'{alpha}', {})
                                                .get(f'{enc_out_dec_inp}', {})
                                                .get(f'{train_snr}', {})
                                                .get(f'{splits}', {})
                                                .get(model, {})
                                                .get("ssim"))
                                    plt.plot(self.SNR_range,np.squeeze(ssim),marker=self.plot_marker.get(f'{model}',{}),ms=10,label=f"{model}/loss Fnc:{lossfu}/beta:{beta}/alpha:{alpha}/bn:{enc_out_dec_inp}/Train SNR:{train_snr}/splits:{splits}")
        plt.xlabel('SNR[dB]', fontsize=16, fontweight='bold')
        plt.ylabel('SSIM', fontsize=16, fontweight='bold')
        plt.grid(True)
        plt.xticks(fontsize=16, fontweight='bold')  
        plt.yticks(fontsize=16, fontweight='bold')  
        # move legend out of the plot
        #plt.legend(prop={'size': 16, 'weight': 'bold'})
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folderpath, f'ssim.png'))
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(os.path.join(folderpath, f'ssim.tex'))
        plt.close()
    
    def accuracy(self,value,folderpath,fig_size=(20,12)):
        plt.figure(figsize=fig_size)
        for lossfu in self.loss_functions:
            for beta in self.betas:
                for alpha in self.alphas:
                    for enc_out_dec_inp in self.enc_out_dec_inps:
                        for train_snr in self.train_snrs:
                            for splits in self.split_image_intos:
                                for model in self.models_to_use:
                                    accuracy = (value.get(f'{lossfu}',{})
                                                .get(f'{beta}',{})
                                                .get(f'{alpha}', {})
                                                .get(f'{enc_out_dec_inp}', {})
                                                .get(f'{train_snr}', {})
                                                .get(f'{splits}', {})
                                                .get(model, {})
                                                .get("accuracy"))
                                    plt.plot(self.SNR_range,np.squeeze(accuracy),marker=self.plot_marker.get(f'{model}',{}),ms=10,label=f"{model}/loss Fnc:{lossfu}/beta:{beta}/alpha:{alpha}/bn:{enc_out_dec_inp}/Train SNR:{train_snr}/splits:{splits}")
        plt.xlabel('SNR[dB]', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
        plt.grid(True)
        plt.xticks(fontsize=16, fontweight='bold')  
        plt.yticks(fontsize=16, fontweight='bold')  
        # move legend out of the plot
        #plt.legend(prop={'size': 16, 'weight': 'bold'})
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folderpath, f'accuracy.png'))
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(os.path.join(folderpath, f'accuracy.tex'))
        plt.close()
    

    def plot_psnr(self,value,folderpath,lossFunction,fig_size=(20,12)):
        plt.figure(figsize=fig_size)
        for beta in self.betas:
            for alpha in self.alphas:
                for enc_out_dec_inp in self.enc_out_dec_inps:
                    for train_snr in self.train_snrs:
                        for splits in self.split_image_intos:
                            for model in self.models_to_use:
                                psnr = (value.get(f'{beta}',{})
                                            .get(f'{alpha}', {})
                                            .get(f'{enc_out_dec_inp}', {})
                                            .get(f'{train_snr}', {})
                                            .get(f'{splits}', {})
                                            .get(model, {})
                                            .get("psnr"))
                                plt.plot(self.SNR_range,np.squeeze(psnr),marker=self.plot_marker.get(f'{model}',{}),ms=10,label=f"{model}/loss Fnc:{lossFunction}/beta:{beta}/alpha:{alpha}/bn:{enc_out_dec_inp}/Train SNR:{train_snr}/splits:{splits}")
        plt.xlabel('SNR[dB]', fontsize=16, fontweight='bold')
        plt.ylabel('PSNR [dB]', fontsize=16, fontweight='bold')
        plt.grid(True)
        plt.xticks(fontsize=16, fontweight='bold')  
        plt.yticks(fontsize=16, fontweight='bold')  
        # move legend out of the plot
        #plt.legend(prop={'size': 16, 'weight': 'bold'})
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folderpath, f'psnr_{lossFunction}.png'))
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(os.path.join(folderpath, f'psnr_{lossFunction}.tex'))
        plt.close()
    
    def plot_ssim(self,value,folderpath,lossFunction,fig_size=(20,12)):
        plt.figure(figsize=fig_size)
        for beta in self.betas:
            for alpha in self.alphas:
                for enc_out_dec_inp in self.enc_out_dec_inps:
                    for train_snr in self.train_snrs:
                        for splits in self.split_image_intos:
                            for model in self.models_to_use:
                                ssim = (value.get(f'{beta}',{})
                                            .get(f'{alpha}', {})
                                            .get(f'{enc_out_dec_inp}', {})
                                            .get(f'{train_snr}', {})
                                            .get(f'{splits}', {})
                                            .get(model, {})
                                            .get("ssim"))
                                plt.plot(self.SNR_range,np.squeeze(ssim),marker=self.plot_marker.get(f'{model}',{}),ms=10,label=f"{model}/loss Fnc:{lossFunction}/beta:{beta}/alpha:{alpha}/bn:{enc_out_dec_inp}/Train SNR:{train_snr}/splits:{splits}")
        plt.xlabel('SNR[dB]', fontsize=16, fontweight='bold')
        plt.ylabel('SSIM', fontsize=16, fontweight='bold')
        plt.grid(True)
        plt.xticks(fontsize=16, fontweight='bold')  
        plt.yticks(fontsize=16, fontweight='bold')  
        # move legend out of the plot
        #plt.legend(prop={'size': 16, 'weight': 'bold'})
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folderpath, f'ssim_{lossFunction}.png'))
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(os.path.join(folderpath, f'ssim_{lossFunction}.tex'))
        plt.close()

    def plot_snr_accuracy_bar_plot(self,result_value,dataset_size,folderPath,lossFunction,fig_size=(20,12)):
        bar_width = 0.8 / len(self.models_to_use)  # Adjust bar width based on number of models
        x = np.arange(len(self.SNR_range))  # Positions for SNR values

        # Loop over all parameters to create the plots
        for beta in self.betas:
            for alpha in self.alphas:
                for enc_values in self.enc_out_dec_inps:
                    for snr_value in self.train_snrs:
                        for split_image in self.split_image_intos:
                            plt.figure(figsize=fig_size)
                            for model_idx,model in enumerate(self.models_to_use):
                                # Configure model-specific attributes
                                if model == 'CNN':
                                    color = "#1f77b4"  # Blue tone
                                elif model == 'ResNet14':
                                    color = "#ff7f0e"  # Orange tone
                                elif model == 'ResNet20':
                                    color = "#2ca02c"  # Green tone
                                elif model == 'VIT':
                                    color = "#9467bd"  # Green tone
                                else:
                                    continue
                                # Position each bar dynamically
                                xPos = x + (model_idx - len(self.models_to_use) / 2) * bar_width
                                accuracy = (result_value.get(f'{beta}', {}).get(f'{alpha}', {})
                                                .get(f'{enc_values}', {})
                                                .get(f'{snr_value}', {})
                                                .get(f'{split_image}', {})
                                                .get(model, {})
                                                .get("accuracy"))
                                bars = plt.bar(
                                    xPos,
                                    accuracy,
                                    width=bar_width,
                                    label=model,  # Add label only once
                                    color=color,
                                    alpha=0.8  # Transparency for layering effect
                                )
                                
                                # Corrected placement of bar_label
                                plt.bar_label(bars, label_type='edge',fontsize=12, rotation=90)
                            # Customize plot appearance
                            plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
                            plt.xlabel('SNR', fontsize=16, fontweight='bold')
                            plt.ylim(0, 1)
                            plt.xticks(x, self.SNR_range, fontsize=14)
                            plt.yticks(fontsize=14)
                            plt.title(f'SNR vs Accuracy \n Split Image: {split_image} | Train SNR : {snr_value:.2f} dB\n'
                                    f'Input Test Data Size: {dataset_size} | Alpha: {alpha}| Beta: {beta} | BN: {enc_values}',
                                    fontsize=10, fontweight='bold')
                            
                            # Legend outside plot
                            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(self.models_to_use), fontsize=10)
                            
                            # Final touches 
                            plt.grid(True, axis='y', linestyle='--', alpha=0.8)
                            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend

                            # Save plot with both PNG and PGF formats
                            os.makedirs(folderPath, exist_ok=True)
                            file_path = f'SNR_vs_Accuracy_splitImage_{split_image}_trainSnr_{snr_value:.2f}_alpha_{alpha}_beta_{beta}_bn_{enc_values}_{lossFunction}'
                            plt.savefig(os.path.join(folderPath, f'{file_path}.png'), dpi=300)
                            tikzplotlib_fix_ncols(plt.gcf())
                            tikzplotlib.save(os.path.join(folderPath, f'{file_path}.tex'))
                            plt.close()

    def dual_psnr_accuracy_with_different_alphas(self, value, folderPath,evaluation_snr_index,fig_size=(20,12)):
        # Create subplots: 2 row, 1 columns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, sharex=True)

        for l in self.loss_functions:
            for beta in self.betas:
                for b in self.enc_out_dec_inps:
                    for t in self.train_snrs:
                        for s in self.split_image_intos:
                            for model in self.models_to_use:
                                model = model
                                rec_values = []
                                class_values = []
                                labels_done = False
                                for a in self.alphas:
                                    psnr_model_history_list = value.get(f"{l}", {}).get(f"{beta}",{}).get(f"{a}",{}).get(f"{b}", {}).get(f"{t}", {}).get(f"{s}", {}).get(model, {}).get('psnr', [])
                                    acc_model_history_list = value.get(f"{l}", {}).get(f"{beta}",{}).get(f"{a}",{}).get(f"{b}", {}).get(f"{t}", {}).get(f"{s}", {}).get(model, {}).get('accuracy', [])

                                    rec_value_list = psnr_model_history_list[evaluation_snr_index]
                                    class_value_list = acc_model_history_list[evaluation_snr_index]

                                    rec_values.append(rec_value_list)
                                    class_values.append(class_value_list)

                                    if not labels_done:
                                        rec_label = f'{model}-PSNR/loss:{l}/BN-{b}/SNR-{t}/splits-{s}/beta-{beta}'
                                        class_label = f'{model}-Acc/loss:{l}/BN-{b}/SNR-{t}/splits-{s}/beta-{beta}'
                                        labels_done = True

                                if rec_values:
                                    ax1.plot(self.alphas[:len(rec_values)], rec_values, marker='o', linestyle='--', label=rec_label)
                                if class_values:
                                    ax2.plot(self.alphas[:len(class_values)], class_values, marker='s', linestyle='-.', label=class_label)

        # Axes settings
        ax1.set_ylabel('PSNR [dB]')
        ax2.set_ylabel('Accuracy')
        
        ax2.set_xlabel('Alpha')
        ax2.grid(True)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_xticks(self.alphas)

        # Common legend at bottom center
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2

        fig.legend(all_handles, all_labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))

        # Final layout adjustments
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.2)  # Adds padding between title and plots, and for legend
        plt.suptitle('PSNR and Accuracy vs Alpha', fontsize=16, y=0.97)
        # Save and close
        plt.savefig(os.path.join(folderPath, f'accuracy_and_psnr_vs_alpha_beta.png'), bbox_inches='tight')
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(os.path.join(folderPath, f'accuracy_and_psnr_vs_alpha_beta.tex'))
        plt.close()

    def three_ssim_psnr_accuracy_with_different_alphas(self, value, folderPath, evaluation_snr_index,
                                                       fig_size=(20, 12)):
        # Create subplots: 2 row, 1 columns
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=fig_size, sharex=True)

        for l in self.loss_functions:
            for beta in self.betas:
                for b in self.enc_out_dec_inps:
                    for t in self.train_snrs:
                        for s in self.split_image_intos:
                            for model in self.models_to_use:
                                model = model
                                rec_values = []
                                class_values = []
                                ssim_values = []
                                labels_done = False
                                for a in self.alphas:
                                    psnr_model_history_list = value.get(f"{l}", {}).get(f"{beta}", {}).get(f"{a}",
                                                                                                           {}).get(
                                        f"{b}", {}).get(f"{t}", {}).get(f"{s}", {}).get(model, {}).get('psnr', [])
                                    ssim_model_history_list = value.get(f"{l}", {}).get(f"{beta}", {}).get(f"{a}",
                                                                                                           {}).get(
                                        f"{b}", {}).get(f"{t}", {}).get(f"{s}", {}).get(model, {}).get('ssim', [])
                                    acc_model_history_list = value.get(f"{l}", {}).get(f"{beta}", {}).get(f"{a}",
                                                                                                          {}).get(
                                        f"{b}", {}).get(f"{t}", {}).get(f"{s}", {}).get(model, {}).get('accuracy', [])

                                    rec_value_list = psnr_model_history_list[evaluation_snr_index]
                                    ssim_value_list = ssim_model_history_list[evaluation_snr_index]
                                    class_value_list = acc_model_history_list[evaluation_snr_index]

                                    rec_values.append(rec_value_list)
                                    class_values.append(class_value_list)
                                    ssim_values.append(ssim_value_list)

                                    if not labels_done:
                                        rec_label = f'{model}-PSNR/loss:{l}/BN-{b}/SNR-{t}/splits-{s}/beta-{beta}'
                                        class_label = f'{model}-Acc/loss:{l}/BN-{b}/SNR-{t}/splits-{s}/beta-{beta}'
                                        ssim_label = f'{model}-SSIM/loss:{l}/BN-{b}/SNR-{t}/splits-{s}/beta-{beta}'
                                        labels_done = True

                                if rec_values:
                                    ax1.plot(self.alphas[:len(rec_values)], rec_values, marker='o', linestyle='--',
                                             label=rec_label)
                                if ssim_values:
                                    ax2.plot(self.alphas[:len(ssim_values)], ssim_values, marker='s', linestyle='-.',
                                             label=ssim_label)
                                if class_values:
                                    ax3.plot(self.alphas[:len(class_values)], class_values, marker='s', linestyle='-.',
                                             label=class_label)

        # Axes settings
        ax1.set_ylabel('PSNR [dB]')
        ax2.set_ylabel('SSIM')
        ax3.set_ylabel('Accuracy')

        ax3.set_xlabel('Alpha')
        ax3.grid(True)
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_xticks(self.alphas)

        # Common legend at bottom center
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()
        all_handles = handles1 + handles2 + handles3
        all_labels = labels1 + labels2 + labels3

        fig.legend(all_handles, all_labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))

        # Final layout adjustments
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.2)  # Adds padding between title and plots, and for legend
        plt.suptitle('PSNR,SSIM and Accuracy vs Alpha', fontsize=16, y=0.97)
        # Save and close
        plt.savefig(os.path.join(folderPath, f'accuracy_SSIM_and_psnr_vs_alpha_beta.png'), bbox_inches='tight')
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(os.path.join(folderPath, f'accuracy_ssim_and_psnr_vs_alpha_beta.tex'))
        plt.close()

    def dual_psnr_ssim_with_different_betas(self, value, folderPath,evaluation_snr_index,fig_size=(20,12)):
        # Create subplots: 2 row, 1 columns
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, sharex=True)

        for l in self.loss_functions:
            for a in self.alphas:
                for b in self.enc_out_dec_inps:
                    for t in self.train_snrs:
                        for s in self.split_image_intos:
                            for model in self.models_to_use:
                                model = model
                                rec_values = []
                                class_values = []
                                labels_done = False
                                for beta in self.betas:
                                    psnr_model_history_list = value.get(f"{l}", {}).get(f"{beta}",{}).get(f"{a}",{}).get(f"{b}", {}).get(f"{t}", {}).get(f"{s}", {}).get(model, {}).get('psnr', [])
                                    ssim_model_history_list = value.get(f"{l}", {}).get(f"{beta}",{}).get(f"{a}",{}).get(f"{b}", {}).get(f"{t}", {}).get(f"{s}", {}).get(model, {}).get('ssim', [])

                                    rec_value_list = psnr_model_history_list[evaluation_snr_index]
                                    class_value_list = ssim_model_history_list[evaluation_snr_index]

                                    rec_values.append(rec_value_list)
                                    class_values.append(class_value_list)

                                    if not labels_done:
                                        rec_label = f'{model}-PSNR/loss:{l}/BN-{b}/SNR-{t}/splits-{s}/beta-{beta}'
                                        class_label = f'{model}-SSIM/loss:{l}/BN-{b}/SNR-{t}/splits-{s}/beta-{beta}'
                                        labels_done = True

                                if rec_values:
                                    ax1.plot(self.betas[:len(rec_values)], rec_values, marker=self.plot_marker.get(f'{model}',{}),ms=10, linestyle='--', label=rec_label)
                                if class_values:
                                    ax2.plot(self.betas[:len(class_values)], class_values, marker=self.plot_marker.get(f'{model}',{}),ms=10, linestyle='-.', label=class_label)

        # Axes settings
        ax1.set_ylabel('PSNR [dB]')
        ax2.set_ylabel('SSIM')
        
        ax2.set_xlabel('Alpha')
        ax2.grid(True)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_xticks(self.betas)

        # Common legend at bottom center
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        all_handles = handles1 + handles2
        all_labels = labels1 + labels2

        fig.legend(all_handles, all_labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))

        # Final layout adjustments
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.2)  # Adds padding between title and plots, and for legend
        plt.suptitle('PSNR and Accuracy vs Alpha', fontsize=16, y=0.97)
        # Save and close
        plt.savefig(os.path.join(folderPath, f'ssim_and_psnr_vs_betas.png'), bbox_inches='tight')
        tikzplotlib_fix_ncols(plt.gcf())
        tikzplotlib.save(os.path.join(folderPath, f'ssim_and_psnr_vs_beta.tex'))
        plt.close()