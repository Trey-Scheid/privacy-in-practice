import pandas as pd
import numpy as np
from scipy.stats import zscore
import pickle
import os 

class DevicePreprocessor:
    def __init__(self,parquet_file,output_parquet_path,count_parameter, limit=1000000):
        self.output_parquet_path=output_parquet_path
        self.parquet_file = parquet_file
        self.limit = limit
        self.data = None
        self.filtered_data = None
        self.sw_category_counts = None
        self.valid_devices = None
        self.weekwise_data = None
        self.l1_distances = None
        self.count_parameter=count_parameter

    def load_data(self):
        """Load data from the Parquet file into a Pandas DataFrame."""
        self.data = pd.read_parquet(self.parquet_file)


    def extract_year_week(self):
        """Extract year and week information from `interval_start_utc`."""
        self.data["year_week"] = self.data["interval_start_utc"].dt.strftime('%Y-%U')

    def count_cat_name_per_device(self):
        """Count the occurrences of `sw_category` per `guid`."""
        
    
        software_categories = {
            'Gaming (Casual, Online & Offline)': ['game.exe', 'roblox.exe', 'steam.exe', 'epic.exe', 'league.exe', 'valorant.exe', 'minecraft.exe', 'wow.exe', 'blizzard.exe', 'origin.exe', 'uplay.exe', 'battle.net.exe', 'genshin.exe', 'pubg.exe', 'fortnite.exe', 'csgo.exe', 'dota.exe', 'hearthstone.exe', 'runelite.exe', 'gacha.exe'],
            
            'Multimedia Editing (Audio & Video)': ['photoshop.exe', 'premiere.exe', 'afterfx.exe', 'illustrator.exe', 'audacity.exe', 'vlc.exe', 'vegas.exe', 'lightroom.exe', 'indesign.exe', 'audition.exe', 'obs.exe', 'davinci.exe', 'krita.exe', 'gimp.exe', 'clipstudio.exe', 'fl studio.exe', 'ableton.exe', 'media.exe', 'player.exe'],
            
            'Development & Programming (IDEs, Text Editors, Version Control)': ['code.exe', 'visual studio.exe', 'pycharm.exe', 'eclipse.exe', 'intellij.exe', 'webstorm.exe', 'android studio.exe', 'xcode.exe', 'sublime.exe', 'atom.exe', 'notepad++.exe', 'git.exe', 'github.exe', 'gitlab.exe', 'sourcetree.exe', 'vim.exe', 'emacs.exe', 'matlab.exe', 'rstudio.exe', 'jupyter.exe'],
            
            'Simulation & Virtual Reality': ['vr.exe', 'simulation.exe', 'sandbox.exe', 'unity.exe', 'unreal.exe', 'fusion360.exe', 'revit.exe', '3dsmax.exe', 'maya.exe', 'blender.exe', 'cad.exe', 'solidworks.exe', 'autocad.exe', 'inventor.exe'],
            
            'Productivity & Office': ['excel.exe', 'word.exe', 'powerpoint.exe', 'outlook.exe', 'onenote.exe', 'teams.exe', 'slack.exe', 'zoom.exe', 'skype.exe', 'discord.exe', 'notion.exe', 'evernote.exe', 'pdf.exe'],
            
            'Web Browsers & Communication': ['chrome.exe', 'firefox.exe', 'edge.exe', 'opera.exe', 'safari.exe', 'brave.exe', 'vivaldi.exe', 'telegram.exe', 'whatsapp.exe', 'wechat.exe', 'messenger.exe'],
            
            'System & Utilities': ['cmd.exe', 'powershell.exe', 'task.exe', 'control.exe', 'registry.exe', 'device.exe', 'service.exe', 'config.exe', 'setup.exe', 'install.exe', 'update.exe', 'driver.exe'],
            
            'Security & Network': ['antivirus.exe', 'firewall.exe', 'vpn.exe', 'proxy.exe', 'security.exe', 'defender.exe', 'norton.exe', 'mcafee.exe', 'avast.exe', 'kaspersky.exe'],
            
            'Other': []
        }

        # Reverse mapping: .exe -> category
        exe_to_category = {exe: category for category, exe_list in software_categories.items() for exe in exe_list}

        # Map each process name to its category
        self.data['sw_category'] = self.data['proc_name'].map(lambda x: exe_to_category.get(x, "Other"))
        self.sw_category_counts = self.data.groupby(["guid", "year_week", "sw_category"]).size().reset_index(name="count")

    def filter_valid_devices(self):
        """Filter devices where `sw_category_name` count is greater than a threshold."""
        device_counts = self.sw_category_counts.groupby("guid")["count"].sum().reset_index()
        self.valid_devices = device_counts[device_counts["count"] > self.count_parameter]

    def merge_filtered_data(self):
        """Merge filtered devices with the original category count data."""
        self.filtered_data = self.sw_category_counts[self.sw_category_counts["guid"].isin(self.valid_devices["guid"])]

    def standardize_usage_counts(self):
        """Standardize category usage counts within each device using Z-score."""
        self.filtered_data["zscore_count"] = self.filtered_data.groupby("guid")["count"].transform(zscore)

    def compute_l1_distances(self):
        """Compute L1 distance for standardized category counts across consecutive weeks per device."""
        self.weekwise_data = self.filtered_data.pivot_table(index=["guid", "sw_category"], columns="year_week", values="zscore_count", aggfunc="sum").fillna(0)
        self.l1_distances = self.weekwise_data.diff(axis=1).abs().sum(axis=1).reset_index()
        self.l1_distances = self.l1_distances.groupby("guid")[0].sum().reset_index()
        self.l1_distances.columns = ['guid', 'l1_distance']

    def save(self):
        """Save the loaded data into a Parquet file."""
        if self.data is not None:
            if self.l1_distances is not None:
                self.l1_distances.to_parquet(self.output_parquet_path, index=False)
        else:
            print("No data to save. Make sure to process the data first.")
    def preprocess(self):
        """Run all preprocessing steps."""
        self.load_data()
        self.extract_year_week()
        self.count_cat_name_per_device()
        self.filter_valid_devices()
        self.merge_filtered_data()
        self.standardize_usage_counts()
        self.compute_l1_distances()
        self.save()


