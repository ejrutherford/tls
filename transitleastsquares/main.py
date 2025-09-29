from __future__ import division, print_function
from os import path
import multiprocessing
import numpy
import sys
import warnings
from functools import partial
from tqdm import tqdm

# TLS parts
from transitleastsquares.results import transitleastsquaresresults
import transitleastsquares.tls_constants as tls_constants
from transitleastsquares.stats import (
    FAP,
    rp_rs_from_depth,
    period_uncertainty,
    spectra,
    final_T0_fit,
    model_lightcurve,
    all_transit_times,
    calculate_transit_duration_in_days,
    calculate_stretch,
    calculate_fill_factor,
    intransit_stats,
    snr_stats,
    count_stats,
)
from transitleastsquares.catalog import catalog_info
from transitleastsquares.helpers import resample, transit_mask
from transitleastsquares.helpers import impact_to_inclination
from transitleastsquares.grid import duration_grid, period_grid
from transitleastsquares.core import (
    edge_effect_correction,
    lowest_residuals_in_this_duration,
    out_of_transit_residuals,
    fold,
    foldfast,
    search_period,
)
from transitleastsquares.transit import reference_transit, fractional_transit, get_cache
from transitleastsquares.validate import validate_inputs, validate_args


class transitleastsquares(object):
    """Compute the transit least squares of limb-darkened transit models"""

    def __init__(self, t, y, dy=None, verbose=True):
        self.t, self.y, self.dy = validate_inputs(t, y, dy)
        self.verbose = verbose

    def power(self, **kwargs):
        """Compute the periodogram for a set of user-defined parameters"""
        self, kwargs = validate_args(self, kwargs)

        if self.verbose:
            print(tls_constants.TLS_VERSION)

        # Added to reduce runtime for sparsely sampled data, e.g. TESS sectors well-separated in time
        gaps = (numpy.diff(self.t)) > 5.0)
        gaps = numpy.where(gaps > 5.0, gaps, 0)
        time_span = numpy.max(self.t) - numpy.min(self.t) - numpy.sum(gaps)
        
        periods = period_grid(
            R_star=self.R_star,
            M_star=self.M_star,
            time_span=time_span,
            period_min=self.period_min,
            period_max=self.period_max,
            oversampling_factor=self.oversampling_factor,
            n_transits_min=self.n_transits_min,
        )

        durations = duration_grid(
            periods, shortest=1 / len(self.t), log_step=self.duration_grid_step
        )

        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        lc_cache_overview, lc_arr = get_cache(
            durations=durations,
            maxwidth_in_samples=maxwidth_in_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark,
            verbose=self.verbose
        )

        if self.verbose:
            print(
                "Searching "
                + str(len(self.y))
                + " data points, "
                + str(len(periods))
                + " periods from "
                + str(round(min(periods), 3))
                + " to "
                + str(round(max(periods), 3))
                + " days"
            )

        # Python 2 multiprocessing with "partial" doesn't work
        # For now, only single-threading in Python 2 is supported
        if sys.version_info[0] < 3:
            self.use_threads = 1
            warnings.warn("This TLS version supports no multithreading on Python 2")

        if self.verbose:
            if self.use_threads == multiprocessing.cpu_count():
                print("Using all " + str(self.use_threads) + " CPU threads")
            else:
                print(
                    "Using "
                    + str(self.use_threads)
                    + " of "
                    + str(multiprocessing.cpu_count())
                    + " CPU threads"
                )

        if self.show_progress_bar:
            bar_format = "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} periods | {elapsed}<{remaining}"
            pbar = tqdm(total=numpy.size(periods), smoothing=0.3, bar_format=bar_format)

        if tls_constants.PERIODS_SEARCH_ORDER == "ascending":
            periods = reversed(periods)
        elif tls_constants.PERIODS_SEARCH_ORDER == "descending":
            pass  # it already is
        elif tls_constants.PERIODS_SEARCH_ORDER == "shuffled":
            periods = numpy.random.permutation(periods)
        else:
            raise ValueError("Unknown PERIODS_SEARCH_ORDER")

        # Result lists now (faster), convert to numpy array later
        test_statistic_periods = []
        test_statistic_residuals = []
        test_statistic_rows = []
        test_statistic_depths = []

        if self.use_threads > 1:  # Run multi-core search
            pool = multiprocessing.Pool(processes=self.use_threads)
            params = partial(
                search_period,
                t=self.t,
                y=self.y,
                dy=self.dy,
                transit_depth_min=self.transit_depth_min,
                R_star_min=self.R_star_min,
                R_star_max=self.R_star_max,
                M_star_min=self.M_star_min,
                M_star_max=self.M_star_max,
                lc_arr=lc_arr,
                lc_cache_overview=lc_cache_overview,
                T0_fit_margin=self.T0_fit_margin,
            )
            for data in pool.imap_unordered(params, periods):
                test_statistic_periods.append(data[0])
                test_statistic_residuals.append(data[1])
                test_statistic_rows.append(data[2])
                test_statistic_depths.append(data[3])
                if self.show_progress_bar:
                    pbar.update(1)
            pool.close()
        else:
            for period in periods:
                data = search_period(
                    period=period,
                    t=self.t,
                    y=self.y,
                    dy=self.dy,
                    transit_depth_min=self.transit_depth_min,
                    R_star_min=self.R_star_min,
                    R_star_max=self.R_star_max,
                    M_star_min=self.M_star_min,
                    M_star_max=self.M_star_max,
                    lc_arr=lc_arr,
                    lc_cache_overview=lc_cache_overview,
                    T0_fit_margin=self.T0_fit_margin,
                )
                test_statistic_periods.append(data[0])
                test_statistic_residuals.append(data[1])
                test_statistic_rows.append(data[2])
                test_statistic_depths.append(data[3])
                if self.show_progress_bar:
                    pbar.update(1)

        if self.show_progress_bar:
            pbar.close()

        # imap_unordered delivers results in unsorted order ==> sort
        test_statistic_periods = numpy.array(test_statistic_periods)
        sort_index = numpy.argsort(test_statistic_periods)
        test_statistic_periods = test_statistic_periods[sort_index]
        test_statistic_residuals = numpy.array(test_statistic_residuals)[sort_index]
        test_statistic_rows = numpy.array(test_statistic_rows)[sort_index]
        test_statistic_depths = numpy.array(test_statistic_depths)[sort_index]

        ### Significant changes made to code below this point to use this
        ### implementation for injection-recovery tests
        ### Goals were to search for a T0 value for peak periods in the
        ### periodpogram. 
        ### Calculation of SNR, odd-even mismatch, empty transit counts,
        ### FAP, in-out of transit counts, etc
        ### not necessary for injection-recovery tests
        
        if max(test_statistic_residuals) == min(test_statistic_residuals):
            no_transits_were_fit = True
            warnings.warn('No transit were fit. Try smaller "transit_depth_min"')
            return "No transits were fit"
        else:
            maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.t))
    
            # Power spectra variants
            chi2 = test_statistic_residuals
            # Additional variants are unneccessary, and are commented out.
            #degrees_of_freedom = 4
            #chi2red = test_statistic_residuals / (len(self.t) - degrees_of_freedom)

            SR, power_raw, power, SDE_raw, SDE = spectra(chi2, self.oversampling_factor)
    
            import scipy
            
            # Initialise empty list for results to be appended to
            results_list = []
            
            # Identify indices at which peaks in the power spectrum occur
            maxima_indices = scipy.signal.argrelextrema(power, numpy.greater)
            
            # Select for peaks with a power (SDE) greater than 3 (arbitrary value)
            # In the tests, marginally significant detections require SDE > 5
            high_sdes = numpy.argwhere(power > 3.0)
            high_sdes = numpy.concatenate(high_sdes)
            intersect = numpy.intersect1d(maxima_indices, high_sdes)            
            maxima_indices = intersect[numpy.argsort(power[intersect])[-10:]]
            
            # Occasionally, the peak value was not included with the above code.
            # Added this to ensure highest value was in the list
            index_highest_power = numpy.argmax(power)
            if index_highest_power not in maxima_indices:
                numpy.append(maxima_indices, index_highest_power)
            
            # For each (with max number of 10) of the highest peaks in the power 
            # spectrum, calculate the period, power, t0, duration and depth
            for index in maxima_indices:
                best_row = test_statistic_rows[index]
                duration = lc_cache_overview["duration"][best_row]
                period = test_statistic_periods[index]
                depth = test_statistic_depths[index]
                power_value = power[index]
                
                T0 = final_T0_fit(
                    signal=lc_arr[best_row],
                    depth=depth,
                    t=self.t,
                    y=self.y,
                    dy=self.dy,
                    period=period,
                    T0_fit_margin=self.T0_fit_margin,
                    show_progress_bar=self.show_progress_bar,
                    verbose=self.verbose
                )
                
                # Generate the model light curve for the above parameters
                
                transit_times = all_transit_times(T0, self.t, period)
    
                transit_duration_in_days = calculate_transit_duration_in_days(
                    self.t, period, transit_times, duration
                )
                duration = transit_duration_in_days
    
                # Folded model / model curve
                # Data phase 0.5 is not always at the midpoint (not at cadence: len(y)/2),
                # so we need to roll the model to match the model so that its mid-transit
                # is at phase=0.5
                stretch = calculate_stretch(self.t, period, transit_times)
                internal_samples = (
                    int(len(self.y) / len(transit_times))
                ) * tls_constants.OVERSAMPLE_MODEL_LIGHT_CURVE
    
                # Full unfolded light curve model
                model_transit_single = fractional_transit(
                    duration=(duration * maxwidth_in_samples),
                    maxwidth=maxwidth_in_samples / stretch,
                    depth=1 - depth,
                    samples=internal_samples,
                    per=self.per,
                    rp=self.rp,
                    a=self.a,
                    inc=self.inc,
                    ecc=self.ecc,
                    w=self.w,
                    u=self.u,
                    limb_dark=self.limb_dark,
                )
                model_lightcurve_model, model_lightcurve_time = model_lightcurve(
                    transit_times, period, self.t, model_transit_single
                )
    
                results_list.append(
                    dict(period=period,
                         power=power_value,
                         T0=T0,
                         duration=duration,
                         depth=depth,
                         model_time=model_lightcurve_time,
                         model_flux=model_lightcurve_model
                         )
                    )
            # return the power spectrum (period, power) and the parameters
            # for the (max 10) highest peaks in the power spectrum
            return test_statistic_periods, power, results_list

